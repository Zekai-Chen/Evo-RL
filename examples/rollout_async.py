#!/usr/bin/env python
"""
Rollout with intervention using async remote inference.

Combines lerobot-human-inloop-record's recording + intervention with
remote async policy inference via gRPC PolicyServer.

No model is loaded locally — all inference happens on the remote server.
No CUDA required on the local machine.

Usage:
    python examples/rollout_async.py \
        --server=192.168.1.73:8080 \
        --policy_type=pi05 \
        --policy_path=Zekai-Chen/pi05_fold_towel \
        --num_episodes=50
"""

import argparse
import logging
import sys

from lerobot.async_inference.remote_policy import RemotePolicy, RemotePolicyConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.utils import init_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Async rollout with intervention")
    parser.add_argument("--server", default="192.168.1.73:8080", help="Policy server address")
    parser.add_argument("--policy_type", default="pi05")
    parser.add_argument("--policy_path", default="Zekai-Chen/pi05_fold_towel")
    parser.add_argument("--policy_device", default="cuda", help="Device on server")
    parser.add_argument("--actions_per_chunk", type=int, default=50)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--episode_time", type=int, default=180)
    parser.add_argument("--reset_time", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dataset_repo", default="Zekai-Chen/eval_bi_so101_rollout_4cam")
    parser.add_argument("--task", default="Fold the towel")
    parser.add_argument("--fresh", action="store_true", help="Clear old data")
    args = parser.parse_args()

    init_logging()

    # Build the lerobot-human-inloop-record command args and run it
    # with our RemotePolicy injected
    cmd_args = [
        "lerobot-human-inloop-record",
        "--robot.type=bi_so_follower",
        "--robot.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032350-if00",
        "--robot.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079934-if00",
        "--robot.id=bi_follower",
        '--robot.left_arm_config.cameras={ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1.1:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, top: {type: intelrealsense, serial_number_or_name: "827112073332", width: 640, height: 480, fps: 30, warmup_s: 10}}',
        '--robot.right_arm_config.cameras={ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1.2:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "215322072684", width: 640, height: 480, fps: 30, warmup_s: 10}}',
        "--teleop.type=bi_so_leader",
        "--teleop.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110746-if00",
        "--teleop.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032215-if00",
        "--teleop.id=bi_leader",
        f"--dataset.repo_id={args.dataset_repo}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={args.episode_time}",
        f"--dataset.reset_time_s={args.reset_time}",
        f"--dataset.fps={args.fps}",
        "--dataset.push_to_hub=false",
        "--display_data=true",
        # Use a dummy policy path — we'll replace the policy object
        f"--policy.path={args.policy_path}",
        "--policy.compile_model=false",
        "--policy.device=cpu",
    ]

    # Override sys.argv so draccus parses our args
    sys.argv = cmd_args

    # Import the record function
    from lerobot.scripts.lerobot_human_inloop_record import human_inloop_record

    from lerobot.policies import factory as policy_factory

    _original_make_policy = policy_factory.make_policy

    def _make_remote_policy(policy_cfg, ds_meta=None):
        """Replace local policy with RemotePolicy."""
        # Get lerobot features from ds_meta
        lerobot_features = {}
        if ds_meta is not None:
            for key, feat in ds_meta.features.items():
                if key.startswith("observation."):
                    lerobot_features[key] = feat

        config = RemotePolicyConfig(
            server_address=args.server,
            remote_policy_type=args.policy_type,
            pretrained_path=args.policy_path,
            policy_device=args.policy_device,
            actions_per_chunk=args.actions_per_chunk,
            device="cpu",
        )
        # Copy input/output features from the original policy config
        config.input_features = policy_cfg.input_features
        config.output_features = policy_cfg.output_features

        return RemotePolicy(config, lerobot_features)

    # Monkey-patch
    policy_factory.make_policy = _make_remote_policy

    try:
        human_inloop_record()
    finally:
        policy_factory.make_policy = _original_make_policy


if __name__ == "__main__":
    main()
