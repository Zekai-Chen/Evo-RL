"""
RemotePolicy: async gRPC wrapper with ActionQueue for RTC-compatible inference.

Uses a background thread for inference so the recording loop never blocks.
Actions are consumed from an ActionQueue while the next chunk is being computed.
RTC prev_chunk_left_over is properly tracked via ActionQueue.get_left_over().
"""

import logging
import pickle  # nosec
import threading
import time
from dataclasses import dataclass, field

import grpc
import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.async_inference.helpers import (
    RemotePolicyConfig as _RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)

logger = logging.getLogger(__name__)


@dataclass
class RemotePolicyConfig:
    """Config for RemotePolicy that connects to a gRPC PolicyServer."""
    type: str = "remote"
    server_address: str = "192.168.1.73:8080"
    remote_policy_type: str = "pi05"
    pretrained_path: str = "Zekai-Chen/pi05_fold_towel"
    policy_device: str = "cuda"
    actions_per_chunk: int = 50
    device: str = "cpu"
    use_amp: bool = False
    n_action_steps: int = 50
    chunk_size: int = 50
    compile_model: bool = False
    input_features: dict = field(default_factory=dict)
    output_features: dict = field(default_factory=dict)


class RemotePolicy:
    """Async policy that forwards inference to a remote gRPC PolicyServer.

    Uses a background inference thread and ActionQueue for smooth action execution.
    Compatible with RTC when the server has RTC enabled.

    The recording loop calls select_action() which returns immediately from the
    ActionQueue. A background thread continuously requests new action chunks
    before the queue runs out, passing prev_chunk_left_over for RTC continuity.
    """

    name = "remote"

    def __init__(self, config: RemotePolicyConfig, lerobot_features: dict):
        self.config = config
        self._lerobot_features = lerobot_features
        self._raw_obs = None  # Set by recording_loop before select_action
        self._task = None
        self._timestep = 0
        self._needs_fresh_obs = True  # Discard stale actions after reset

        # ActionQueue for RTC-compatible action buffering
        rtc_config = RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=10.0)
        self._action_queue = ActionQueue(rtc_config)
        self._latency_tracker = LatencyTracker()

        # Thread synchronization
        self._lock = threading.Lock()
        self._new_obs_event = threading.Event()
        self._shutdown = threading.Event()
        self._first_actions_ready = threading.Event()

        # How many actions left before requesting new chunk
        # Should be >= execution_horizon so RTC has enough overlap
        self._refill_threshold = 20

        # Connect to gRPC server
        self.channel = grpc.insecure_channel(
            config.server_address,
            grpc_channel_options(initial_backoff="0.033s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        # Handshake
        logger.info(f"Connecting to policy server at {config.server_address}...")
        self.stub.Ready(services_pb2.Empty())
        logger.info("Connected. Sending policy instructions...")

        # Send policy instructions — server loads the model
        policy_specs = _RemotePolicyConfig(
            policy_type=config.remote_policy_type,
            pretrained_name_or_path=config.pretrained_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=config.actions_per_chunk,
            device=config.policy_device,
        )
        self.stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(policy_specs))
        )
        logger.info("Policy loading on server (this may take a minute)...")

        # Start background inference thread
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="RemotePolicy-Inference"
        )
        self._inference_thread.start()

    def reset(self):
        """Reset action queue between episodes."""
        with self._lock:
            self._action_queue = ActionQueue(
                RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=10.0)
            )
            self._timestep = 0
            self._raw_obs = None  # Prevent background thread from using stale observation
            self._first_actions_ready.clear()
            self._latency_tracker.reset()
            self._new_obs_event.clear()  # Stop background thread from requesting
            self._needs_fresh_obs = True  # Flag to discard stale actions on next select_action

    def select_action(self, observation: dict, task: str | None = None) -> torch.Tensor:
        """Get next action from the queue. Non-blocking after first chunk.

        Called by the recording loop on every frame. Stores the latest observation
        for the background thread to use when requesting the next chunk.
        """
        # After reset, discard any stale actions that the background thread may have queued
        if self._needs_fresh_obs:
            with self._lock:
                self._action_queue = ActionQueue(
                    RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=10.0)
                )
                self._first_actions_ready.clear()
            self._needs_fresh_obs = False

        # Store latest observation for background thread
        with self._lock:
            self._raw_obs = observation
            self._task = task

        # Signal background thread that new observation is available
        self._new_obs_event.set()

        # Wait for first actions (only blocks on very first call)
        if not self._first_actions_ready.is_set():
            self._first_actions_ready.wait(timeout=60.0)

        # Get action from queue
        action = self._action_queue.get()
        if action is not None:
            return action.unsqueeze(0) if action.ndim == 1 else action

        # Queue empty — this means inference is slower than execution.
        # Block and wait for next chunk.
        logger.warning("Action queue empty, waiting for inference...")
        self._new_obs_event.set()  # Ensure inference thread is working
        for _ in range(100):  # Wait up to 10 seconds
            time.sleep(0.1)
            action = self._action_queue.get()
            if action is not None:
                return action.unsqueeze(0) if action.ndim == 1 else action

        logger.error("Timed out waiting for actions")
        return torch.zeros(1, 12)

    def _inference_loop(self):
        """Background thread: continuously requests new action chunks."""
        logger.info("[INFERENCE] Background inference thread started")

        while not self._shutdown.is_set():
            # Wait until we have an observation and queue needs refill
            self._new_obs_event.wait(timeout=1.0)

            queue_size = self._action_queue.qsize()

            # Only request new chunk if queue is running low
            if queue_size > self._refill_threshold and self._first_actions_ready.is_set():
                time.sleep(0.01)  # Don't busy-wait
                continue

            # Get current observation
            with self._lock:
                obs = self._raw_obs
                task = self._task

            if obs is None:
                continue

            self._new_obs_event.clear()

            try:
                start_time = time.perf_counter()

                # Get leftover actions for RTC
                prev_actions = self._action_queue.get_left_over()
                action_index_before = self._action_queue.get_action_index()

                # Build observation for server
                raw_obs = dict(obs)
                if task is not None:
                    raw_obs["task"] = task
                # Tell server how many actions remain so it can compute correct RTC offset
                raw_obs["__rtc_queue_remaining__"] = self._action_queue.qsize()

                timed_obs = TimedObservation(
                    timestamp=time.time(),
                    observation=raw_obs,
                    timestep=self._timestep,
                )
                timed_obs.must_go = True

                # Send observation to server
                obs_bytes = pickle.dumps(timed_obs)
                obs_iterator = send_bytes_in_chunks(
                    obs_bytes,
                    services_pb2.Observation,
                    log_prefix="[INFERENCE] Obs",
                    silent=True,
                )
                self.stub.SendObservations(obs_iterator)

                # Get action chunk from server (blocks until inference done)
                actions_response = self.stub.GetActions(services_pb2.Empty())

                if len(actions_response.data) == 0:
                    logger.warning("[INFERENCE] Server returned empty actions")
                    continue

                timed_actions: list[TimedAction] = pickle.loads(actions_response.data)  # nosec

                if not timed_actions:
                    continue

                # Extract action tensors
                action_tensors = torch.stack([ta.get_action().cpu() for ta in timed_actions])

                # Calculate real inference delay
                inference_time = time.perf_counter() - start_time
                self._latency_tracker.add(inference_time)
                time_per_step = 1.0 / 30  # fps
                real_delay = max(0, int(inference_time / time_per_step))

                # Merge into ActionQueue with RTC tracking
                self._action_queue.merge(
                    action_tensors,       # original actions (for RTC left_over)
                    action_tensors,       # processed actions (for execution)
                    real_delay,
                    action_index_before,
                )

                self._timestep += len(timed_actions)
                self._first_actions_ready.set()

                logger.debug(
                    f"[INFERENCE] Chunk received: {len(timed_actions)} actions, "
                    f"inference={inference_time:.3f}s, delay={real_delay}, "
                    f"queue={self._action_queue.qsize()}"
                )

            except Exception as e:
                logger.error(f"[INFERENCE] Error: {e}")
                import traceback
                traceback.print_exc()

        logger.info("[INFERENCE] Background inference thread stopped")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("RemotePolicy does not support forward()")

    def __del__(self):
        self._shutdown.set()
        if hasattr(self, "_inference_thread") and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5.0)
        if hasattr(self, "channel"):
            self.channel.close()
