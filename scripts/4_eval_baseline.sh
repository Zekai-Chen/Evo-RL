#!/bin/bash
# ============================================
# Step 4a: Evaluate baseline policy (no teleop)
# ============================================
# Usage:
#   bash scripts/4_eval_baseline.sh          # Run 5 episodes (default)
#   bash scripts/4_eval_baseline.sh 10       # Run 10 episodes
#
# The policy controls the robot autonomously.
# Press Right Arrow to end an episode early.
# Press Esc to stop the session.

# --- Config ---
POLICY_REPO="Zekai-Chen/pi05_fold_towel"
DATASET_REPO="Zekai-Chen/eval_bi_so101_baseline"
TASK="Fold the towel"
FPS=30
EPISODE_TIME=180
RESET_TIME=30
NUM_EPISODES=${1:-5}

# Clean up stale state
pkill -f rerun 2>/dev/null
rm -rf ~/.cache/huggingface/lerobot/${DATASET_REPO}
sleep 1

echo "=== Evo-RL Baseline Evaluation ==="
echo "Policy:   ${POLICY_REPO}"
echo "Episodes: ${NUM_EPISODES}"
echo "Time/ep:  ${EPISODE_TIME}s (3 min)"
echo "==================================="

lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032350-if00 \
  --robot.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079934-if00 \
  --robot.id=bi_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1.1:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, top: {type: intelrealsense, serial_number_or_name: "827112073332", width: 640, height: 480, fps: 30, warmup_s: 10}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1.2:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "215322072684", width: 640, height: 480, fps: 30, warmup_s: 10}}' \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes=${NUM_EPISODES} \
  --dataset.episode_time_s=${EPISODE_TIME} \
  --dataset.reset_time_s=${RESET_TIME} \
  --dataset.fps=${FPS} \
  --display_data=true \
  --policy.path=${POLICY_REPO} \
  --policy.compile_model=false \
  --intervention_state_machine_enabled=false
