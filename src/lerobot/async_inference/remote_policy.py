"""
RemotePolicy: a thin wrapper that forwards inference to a remote gRPC PolicyServer.

Integrates with lerobot-human-inloop-record by replacing the local policy.
All inference (preprocessing, model, postprocessing) happens on the remote server.
Locally, no CUDA is needed.
"""

import logging
import pickle  # nosec
import time
from collections import deque
from dataclasses import dataclass

import grpc
import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.async_inference.helpers import (
    RemotePolicyConfig as _RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)

logger = logging.getLogger(__name__)


@dataclass
class RemotePolicyConfig(PreTrainedConfig):
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
    gradient_checkpointing: bool = False


class RemotePolicy:
    """Policy that forwards inference to a remote gRPC PolicyServer.

    Drop-in replacement for PreTrainedPolicy in the recording pipeline.
    No model is loaded locally — all inference happens on the remote server.
    """

    name = "remote"

    def __init__(self, config: RemotePolicyConfig, lerobot_features: dict):
        self.config = config
        self._action_queue = deque()
        self._lerobot_features = lerobot_features
        self._timestep = 0

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

    def reset(self):
        """Reset action queue between episodes."""
        self._action_queue.clear()
        self._timestep = 0

    def select_action(self, observation: dict, task: str | None = None) -> torch.Tensor:
        """Send raw observation to server, get postprocessed action back.

        Args:
            observation: dict of numpy arrays from the recording loop (raw robot observation)
            task: task string for the policy

        Returns:
            Tensor of shape (1, action_dim) — same format as local policy output
        """
        # Return buffered action if available
        if len(self._action_queue) > 0:
            action = self._action_queue.popleft()
            return action.unsqueeze(0) if action.ndim == 1 else action

        # Build observation for server — add task
        raw_obs = dict(observation)
        if task is not None:
            raw_obs["task"] = task

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
            log_prefix="[REMOTE] Obs",
            silent=True,
        )
        self.stub.SendObservations(obs_iterator)

        # Get action chunk from server (blocks until server responds)
        actions_response = self.stub.GetActions(services_pb2.Empty())
        if len(actions_response.data) == 0:
            logger.warning("Server returned empty actions, retrying...")
            # Retry once
            actions_response = self.stub.GetActions(services_pb2.Empty())
            if len(actions_response.data) == 0:
                logger.error("Server returned empty actions twice")
                return torch.zeros(1, 12)

        timed_actions: list[TimedAction] = pickle.loads(actions_response.data)  # nosec

        # Buffer action tensors
        for ta in timed_actions:
            self._action_queue.append(ta.get_action().cpu())

        self._timestep += len(timed_actions)

        if len(self._action_queue) > 0:
            action = self._action_queue.popleft()
            return action.unsqueeze(0) if action.ndim == 1 else action
        return torch.zeros(1, 12)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("RemotePolicy does not support forward()")

    def __del__(self):
        if hasattr(self, "channel"):
            self.channel.close()
