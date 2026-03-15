#!/bin/bash
# ============================================
# Step 4a: Evaluate baseline with async inference
# ============================================
# Usage:
#   bash scripts/4_eval_baseline_async.sh                    # Default server 192.168.1.73:8000
#   bash scripts/4_eval_baseline_async.sh 192.168.1.73:8080  # Custom server address
#
# Requires policy server already running on the remote machine.
# Press Ctrl+C to stop.

# --- Config ---
POLICY_REPO="Zekai-Chen/pi05_fold_towel"
TASK="Fold the towel"
SERVER=${1:-"192.168.1.73:8080"}
FPS=30
ACTIONS_PER_CHUNK=50

echo "=== Evo-RL Async Inference + RTC ==="
echo "Server:   ${SERVER}"
echo "Policy:   ${POLICY_REPO}"
echo "FPS:      ${FPS}"
echo "====================================="

python -m lerobot.async_inference.robot_client \
  --server_address=${SERVER} \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032350-if00 \
  --robot.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079934-if00 \
  --robot.id=bi_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1.1:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, top: {type: intelrealsense, serial_number_or_name: "827112073332", width: 640, height: 480, fps: 30, warmup_s: 10}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1.2:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "215322072684", width: 640, height: 480, fps: 30, warmup_s: 10}}' \
  --task="${TASK}" \
  --policy_type=pi05 \
  --pretrained_name_or_path=${POLICY_REPO} \
  --policy_device=cuda \
  --actions_per_chunk=${ACTIONS_PER_CHUNK} \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --fps=${FPS} \
  --debug_visualize_queue_size=True
