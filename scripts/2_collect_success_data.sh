#!/bin/bash
# ============================================
# Step 2: Collect success episodes for baseline
# ============================================
# Usage:
#   bash scripts/2_collect_success_data.sh              # Collect 50 episodes (default)
#   bash scripts/2_collect_success_data.sh 100          # Collect 100 episodes
#   bash scripts/2_collect_success_data.sh 50 --fresh   # Fresh start with 50 episodes
#
# Hotkeys during recording:
#   s           -> Mark SUCCESS and end current episode
#   f           -> Mark FAILURE and end current episode
#   Right Arrow -> End the current loop early
#   Left Arrow  -> End early and re-record current episode
#   Esc         -> Stop the recording session
#
# Tip: Only press 's' when the task is truly completed successfully.
#      Press 'f' to discard bad episodes. Aim for high success rate data.

# --- Config (EDIT THESE for your task) ---
DATASET_REPO="Zekai-Chen/bi_so101_baseline_4cam"
TASK="Fold the towel"
FPS=30
EPISODE_TIME=120
RESET_TIME=30

# --- Argument parsing ---
NUM_EPISODES=${1:-10}
RESUME="true"

for arg in "$@"; do
  if [ "$arg" = "--fresh" ]; then
    echo "Starting fresh (clearing old data)..."
    rm -rf ~/.cache/huggingface/lerobot/${DATASET_REPO}
    RESUME="false"
  fi
done

echo "=== Evo-RL Data Collection ==="
echo "Dataset:  ${DATASET_REPO}"
echo "Task:     ${TASK}"
echo "Episodes: ${NUM_EPISODES}"
echo "Resume:   ${RESUME}"
echo "=============================="

lerobot-human-inloop-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032350-if00 \
  --robot.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079934-if00 \
  --robot.id=bi_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/video17", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, top: {type: intelrealsense, serial_number_or_name: "827112073332", width: 640, height: 480, fps: 30, warmup_s: 5}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/video18", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "215322072684", width: 640, height: 480, fps: 30, warmup_s: 5}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110746-if00 \
  --teleop.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032215-if00 \
  --teleop.id=bi_leader \
  --dataset.repo_id=${DATASET_REPO} \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes=${NUM_EPISODES} \
  --dataset.episode_time_s=${EPISODE_TIME} \
  --dataset.reset_time_s=${RESET_TIME} \
  --dataset.fps=${FPS} \
  --dataset.push_to_hub=true \
  --resume=${RESUME} \
  --display_data=true
