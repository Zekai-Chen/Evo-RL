"""
RemotePolicy: TCP for observations (30ms) + gRPC for actions.
Simple deque buffer. No ActionQueue, no JPEG, no RTC client-side.
"""

import logging
import pickle  # nosec
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import grpc
import torch

from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.async_inference.helpers import (
    RemotePolicyConfig as _RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)

logger = logging.getLogger(__name__)

TCP_OBS_PORT = 9090


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
    """TCP obs + gRPC actions. Simple deque buffer."""

    name = "remote"

    def __init__(self, config: RemotePolicyConfig, lerobot_features: dict):
        self.config = config
        self._lerobot_features = lerobot_features
        self._raw_obs = None
        self._task = None
        self._timestep = 0
        self._chunk_size = config.actions_per_chunk

        # Simple deque
        self._action_deque = deque()
        self._deque_lock = threading.Lock()

        # Threading
        self._lock = threading.Lock()
        self._new_obs_event = threading.Event()
        self._shutdown = threading.Event()
        self._first_actions_ready = threading.Event()

        # Server host
        self._server_host = config.server_address.split(":")[0]

        # gRPC for handshake + GetActions
        self.channel = grpc.insecure_channel(
            config.server_address, grpc_channel_options(initial_backoff="0.033s"),
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
        logger.info("Policy loaded on server.")

        # TCP for observations (bypasses gRPC 800ms overhead)
        self._tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._tcp_sock.connect((self._server_host, TCP_OBS_PORT))
        logger.info(f"TCP connected to {self._server_host}:{TCP_OBS_PORT}")

        # Two threads: send obs (TCP) + receive actions (gRPC)
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._send_thread.start()
        self._receive_thread.start()

    def reset(self):
        with self._deque_lock:
            self._action_deque.clear()
        with self._lock:
            self._timestep = 0
            self._raw_obs = None
            self._first_actions_ready.clear()
            self._new_obs_event.clear()

    def select_action(self, observation: dict, task: str | None = None) -> torch.Tensor:
        with self._lock:
            self._raw_obs = observation
            self._task = task
        self._new_obs_event.set()

        if not self._first_actions_ready.is_set():
            self._first_actions_ready.wait(timeout=60.0)

        with self._deque_lock:
            if self._action_deque:
                action = self._action_deque.popleft()
                return action.unsqueeze(0) if action.ndim == 1 else action

        logger.warning("Action queue empty, waiting...")
        self._new_obs_event.set()
        for _ in range(100):
            time.sleep(0.1)
            with self._deque_lock:
                if self._action_deque:
                    action = self._action_deque.popleft()
                    return action.unsqueeze(0) if action.ndim == 1 else action

        logger.error("Timed out waiting for actions")
        return torch.zeros(1, 12)

    def _send_loop(self):
        """Send observations via TCP. Raw images, no compression."""
        logger.info("[SEND] TCP sender started")

        while not self._shutdown.is_set():
            self._new_obs_event.wait(timeout=1.0)

            # Send when queue below 50% (like robot_client chunk_size_threshold)
            with self._deque_lock:
                qsize = len(self._action_deque)
            if self._first_actions_ready.is_set() and qsize > self._chunk_size * 0.5:
                time.sleep(0.01)
                continue

            with self._lock:
                obs = self._raw_obs
                task = self._task

            if obs is None:
                continue

            self._new_obs_event.clear()

            try:
                raw_obs = dict(obs)
                if task is not None:
                    raw_obs["task"] = task

                timed_obs = TimedObservation(
                    timestamp=time.time(),
                    observation=raw_obs,
                    timestep=self._timestep,
                )
                timed_obs.must_go = True

                t0 = time.perf_counter()
                obs_bytes = pickle.dumps(timed_obs)
                self._tcp_sock.sendall(struct.pack(">I", len(obs_bytes)))
                self._tcp_sock.sendall(obs_bytes)
                send_ms = (time.perf_counter() - t0) * 1000
                logger.debug(f"[SEND] {len(obs_bytes)/1024:.0f}KB in {send_ms:.0f}ms")

            except Exception as e:
                logger.error(f"[SEND] TCP error: {e}")
                try:
                    self._tcp_sock.close()
                    self._tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._tcp_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._tcp_sock.connect((self._server_host, TCP_OBS_PORT))
                except Exception:
                    time.sleep(1)

        logger.info("[SEND] TCP sender stopped")

    def _receive_loop(self):
        """Receive actions via gRPC, append to deque."""
        logger.info("[RECV] gRPC receiver started")

        while not self._shutdown.is_set():
            try:
                t0 = time.perf_counter()
                resp = self.stub.GetActions(services_pb2.Empty())

                if len(resp.data) == 0:
                    continue

                result = pickle.loads(resp.data)  # nosec

                # Handle both dict format and TimedAction format
                if isinstance(result, dict):
                    actions = torch.from_numpy(result["postprocessed"]).float()
                    action_list = [actions[i] for i in range(len(actions))]
                elif isinstance(result, list):
                    action_list = [ta.get_action().cpu() for ta in result]
                else:
                    continue

                get_ms = (time.perf_counter() - t0) * 1000

                with self._deque_lock:
                    for a in action_list:
                        self._action_deque.append(a)
                    qsize = len(self._action_deque)

                self._timestep += len(action_list)
                self._first_actions_ready.set()

                delay = int(get_ms / 1000 * 30)
                logger.warning(f"[TIMING] get={get_ms:.0f}ms delay={delay} queue={qsize}")

            except Exception as e:
                if not self._shutdown.is_set():
                    logger.error(f"[RECV] Error: {e}")
                time.sleep(0.1)

        logger.info("[RECV] gRPC receiver stopped")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __del__(self):
        self._shutdown.set()
        for t in [getattr(self, "_send_thread", None), getattr(self, "_receive_thread", None)]:
            if t and t.is_alive():
                t.join(timeout=5.0)
        if hasattr(self, "_tcp_sock"):
            self._tcp_sock.close()
        if hasattr(self, "channel"):
            self.channel.close()
