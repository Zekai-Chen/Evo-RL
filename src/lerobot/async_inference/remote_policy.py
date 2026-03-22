"""
RemotePolicy: async inference with TCP observation sending + gRPC action receiving.

TCP for observations (3.6MB in ~30ms vs gRPC's 800ms).
gRPC only for handshake + GetActions (~180ms).
Two threads: send (TCP) + receive (gRPC).
"""

import logging
import pickle  # nosec
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import grpc
import numpy as np
import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options
from lerobot.async_inference.helpers import (
    RemotePolicyConfig as _RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)

logger = logging.getLogger(__name__)

TCP_OBS_PORT = 9090  # Dedicated TCP port for observation transfer


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
    """Async remote policy: TCP for observations, gRPC for actions."""

    name = "remote"

    def __init__(self, config: RemotePolicyConfig, lerobot_features: dict):
        self.config = config
        self._lerobot_features = lerobot_features
        self._raw_obs = None
        self._task = None
        self._timestep = 0

        # ActionQueue (append mode)
        self._rtc_config = RTCConfig(enabled=False)
        self._action_queue = ActionQueue(self._rtc_config)
        self._latency_tracker = LatencyTracker()

        self._refill_threshold = 10

        # Threading
        self._lock = threading.Lock()
        self._new_obs_event = threading.Event()
        self._shutdown = threading.Event()
        self._first_actions_ready = threading.Event()

        # Extract server host from address
        self._server_host = config.server_address.split(":")[0]
        self._grpc_port = int(config.server_address.split(":")[1])

        # gRPC channel (only for handshake + GetActions)
        self._grpc_channel = grpc.insecure_channel(
            config.server_address, grpc_channel_options(initial_backoff="0.033s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self._grpc_channel)

        # Handshake via gRPC
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

        # TCP socket for observation sending
        self._tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._tcp_sock.connect((self._server_host, TCP_OBS_PORT))
        logger.info(f"TCP observation channel connected to {self._server_host}:{TCP_OBS_PORT}")

        # Start threads
        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="RemotePolicy-Send"
        )
        self._receive_thread = threading.Thread(
            target=self._receive_loop, daemon=True, name="RemotePolicy-Receive"
        )
        self._send_thread.start()
        self._receive_thread.start()

    def reset(self):
        with self._lock:
            self._action_queue = ActionQueue(self._rtc_config)
            self._timestep = 0
            self._raw_obs = None
            self._first_actions_ready.clear()
            self._latency_tracker.reset()
            self._new_obs_event.clear()

    def select_action(self, observation: dict, task: str | None = None) -> torch.Tensor:
        with self._lock:
            self._raw_obs = observation
            self._task = task
        self._new_obs_event.set()

        if not self._first_actions_ready.is_set():
            self._first_actions_ready.wait(timeout=60.0)

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

    def _send_loop(self):
        """Send observations via TCP (fast, ~30ms for 3.6MB)."""
        logger.info("[SEND] TCP observation sender started")

        while not self._shutdown.is_set():
            self._new_obs_event.wait(timeout=1.0)

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
                # Send: 4-byte length header + payload
                self._tcp_sock.sendall(struct.pack(">I", len(obs_bytes)))
                self._tcp_sock.sendall(obs_bytes)
                send_ms = (time.perf_counter() - t0) * 1000
                logger.debug(f"[SEND] TCP sent {len(obs_bytes)/1024:.0f}KB in {send_ms:.0f}ms")

            except Exception as e:
                logger.error(f"[SEND] TCP error: {e}")
                # Try reconnect
                try:
                    self._tcp_sock.close()
                    self._tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self._tcp_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self._tcp_sock.connect((self._server_host, TCP_OBS_PORT))
                    logger.info("[SEND] TCP reconnected")
                except Exception:
                    time.sleep(1)

        logger.info("[SEND] TCP sender stopped")

    def _receive_loop(self):
        """Receive action chunks via gRPC."""
        logger.info("[RECV] gRPC action receiver started")

        while not self._shutdown.is_set():
            try:
                action_index_before = self._action_queue.get_action_index()
                t0 = time.perf_counter()

                actions_response = self.stub.GetActions(services_pb2.Empty())

                if len(actions_response.data) == 0:
                    continue

                timed_actions: list[TimedAction] = pickle.loads(actions_response.data)  # nosec
                if not timed_actions:
                    continue

                action_tensors = torch.stack([ta.get_action().cpu() for ta in timed_actions])

                inference_time = time.perf_counter() - t0
                self._latency_tracker.add(inference_time)
                time_per_step = 1.0 / 30
                real_delay = max(0, int(inference_time / time_per_step))

                self._action_queue.merge(
                    action_tensors, action_tensors, real_delay, action_index_before,
                )

                self._timestep += len(timed_actions)
                self._first_actions_ready.set()

                logger.warning(
                    f"[TIMING] get={inference_time*1000:.0f}ms delay={real_delay} "
                    f"queue={self._action_queue.qsize()}"
                )

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
        if hasattr(self, "_grpc_channel"):
            self._grpc_channel.close()
