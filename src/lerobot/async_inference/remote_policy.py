"""
RemotePolicy: async gRPC wrapper with proper RTC support.

Following eval_with_real_robot.py reference:
- Background thread handles inference
- ActionQueue stores ORIGINAL (normalized) and POSTPROCESSED actions separately
- get_left_over() returns originals → sent to server as prev_chunk_left_over
- Server returns both original + postprocessed → client merges both into ActionQueue
- On intervention release / episode end: reset_rtc flag clears server RTC state
"""

import logging
import math
import pickle  # nosec
import threading
import time
from dataclasses import dataclass, field

import cv2
import grpc
import numpy as np
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
    name = "remote"

    def __init__(self, config: RemotePolicyConfig, lerobot_features: dict):
        self.config = config
        self._lerobot_features = lerobot_features
        self._raw_obs = None
        self._task = None
        self._timestep = 0
        self._reset_rtc = True  # First chunk has no RTC history

        # RTC config — must match server
        self._rtc_config = RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=10.0)
        self._action_queue = ActionQueue(self._rtc_config)
        self._latency_tracker = LatencyTracker()

        # Reference: action_queue_size_to_get_new_actions should be > execution_horizon + inference_delay
        self._refill_threshold = 30

        # Threading
        self._lock = threading.Lock()
        self._new_obs_event = threading.Event()
        self._shutdown = threading.Event()
        self._first_actions_ready = threading.Event()

        # gRPC
        self.channel = grpc.insecure_channel(
            config.server_address,
            grpc_channel_options(initial_backoff="0.033s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        logger.info(f"Connecting to policy server at {config.server_address}...")
        self.stub.Ready(services_pb2.Empty())
        logger.info("Connected. Sending policy instructions...")

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

        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="RemotePolicy-Inference"
        )
        self._inference_thread.start()

    def reset(self):
        """Called on episode end and intervention release."""
        with self._lock:
            self._action_queue = ActionQueue(self._rtc_config)
            self._timestep = 0
            self._raw_obs = None
            self._first_actions_ready.clear()
            self._latency_tracker.reset()
            self._new_obs_event.clear()
        self._reset_rtc = True

    def select_action(self, observation: dict, task: str | None = None) -> torch.Tensor:
        with self._lock:
            self._raw_obs = observation
            self._task = task
        self._new_obs_event.set()

        if not self._first_actions_ready.is_set():
            self._first_actions_ready.wait(timeout=60.0)

        # ActionQueue.get() returns POSTPROCESSED action
        action = self._action_queue.get()
        if action is not None:
            return action.unsqueeze(0) if action.ndim == 1 else action

        logger.warning("Action queue empty, waiting for inference...")
        self._new_obs_event.set()
        for _ in range(100):
            time.sleep(0.1)
            action = self._action_queue.get()
            if action is not None:
                return action.unsqueeze(0) if action.ndim == 1 else action

        logger.error("Timed out waiting for actions")
        return torch.zeros(1, 12)

    def _inference_loop(self):
        """Background thread following eval_with_real_robot.py pattern."""
        logger.info("[INFERENCE] Background thread started")

        while not self._shutdown.is_set():
            self._new_obs_event.wait(timeout=1.0)

            queue_size = self._action_queue.qsize()
            if queue_size > self._refill_threshold and self._first_actions_ready.is_set():
                time.sleep(0.01)
                continue

            with self._lock:
                obs = self._raw_obs
                task = self._task

            if obs is None:
                continue

            self._new_obs_event.clear()

            try:
                current_time = time.perf_counter()

                # Following reference: get state BEFORE inference
                action_index_before = self._action_queue.get_action_index()
                # get_left_over returns ORIGINAL (normalized) actions
                prev_actions = self._action_queue.get_left_over()

                # Build observation with JPEG compression
                raw_obs = {}
                for k, v in obs.items():
                    if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] == 3:
                        _, enc = cv2.imencode('.jpg', v, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        raw_obs[k] = ("__jpeg__", enc.tobytes())
                    else:
                        raw_obs[k] = v
                if task is not None:
                    raw_obs["task"] = task

                # Send client's RTC leftover (original/normalized) to server
                if prev_actions is not None and not self._reset_rtc:
                    raw_obs["__rtc_prev_chunk_left_over__"] = prev_actions.cpu().numpy()

                if self._reset_rtc:
                    raw_obs["__reset_rtc__"] = True
                    self._reset_rtc = False

                timed_obs = TimedObservation(
                    timestamp=time.time(),
                    observation=raw_obs,
                    timestep=self._timestep,
                )
                timed_obs.must_go = True

                obs_bytes = pickle.dumps(timed_obs)
                obs_iter = send_bytes_in_chunks(
                    obs_bytes, services_pb2.Observation,
                    log_prefix="[INFERENCE] Obs", silent=True,
                )
                self.stub.SendObservations(obs_iter)

                resp = self.stub.GetActions(services_pb2.Empty())
                if len(resp.data) == 0:
                    logger.warning("[INFERENCE] Server returned empty actions")
                    continue

                server_result = pickle.loads(resp.data)  # nosec

                # Server returns dict with original (normalized) + postprocessed
                if isinstance(server_result, dict):
                    original_actions = torch.from_numpy(server_result["original"]).float()
                    postprocessed_actions = torch.from_numpy(server_result["postprocessed"]).float()
                else:
                    # Fallback for old server format (list[TimedAction])
                    postprocessed_actions = torch.stack(
                        [ta.get_action().cpu() for ta in server_result]
                    )
                    original_actions = postprocessed_actions.clone()

                # Following reference: calculate delay and merge
                new_latency = time.perf_counter() - current_time
                time_per_step = 1.0 / 30
                new_delay = math.ceil(new_latency / time_per_step)
                self._latency_tracker.add(new_latency)

                # Merge: ORIGINAL for RTC tracking, POSTPROCESSED for execution
                self._action_queue.merge(
                    original_actions,       # stored in original_queue, returned by get_left_over()
                    postprocessed_actions,   # stored in queue, returned by get()
                    new_delay,
                    action_index_before,
                )

                self._timestep += len(postprocessed_actions)
                self._first_actions_ready.set()

                logger.debug(
                    f"[INFERENCE] Chunk: {len(postprocessed_actions)} actions, "
                    f"latency={new_latency:.3f}s, delay={new_delay}, "
                    f"queue={self._action_queue.qsize()}"
                )

            except Exception as e:
                logger.error(f"[INFERENCE] Error: {e}")
                import traceback
                traceback.print_exc()

        logger.info("[INFERENCE] Background thread stopped")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __del__(self):
        self._shutdown.set()
        if hasattr(self, "_inference_thread") and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=5.0)
        if hasattr(self, "channel"):
            self.channel.close()
