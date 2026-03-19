#!/bin/bash
# ============================================
# Step 4: Rollout with intervention (async inference)
# ============================================
# Usage:
#   bash scripts/4_rollout_async.sh              # 50 episodes (default)
#   bash scripts/4_rollout_async.sh 100          # 100 episodes
#   bash scripts/4_rollout_async.sh --fresh      # Fresh start
#
# Requires policy server running at 192.168.1.73:8080:
#   TORCH_COMPILE_DISABLE=1 python -m lerobot.async_inference.policy_server \
#     --host=0.0.0.0 --port=8080
#
# Hotkeys:
#   i           -> Toggle intervention (policy <-> human)
#   s           -> Mark SUCCESS and end current episode
#   f           -> Mark FAILURE and end current episode
#   Right Arrow -> End the current loop early
#   Esc         -> Stop the recording session

# --- Config ---
export REMOTE_POLICY_SERVER="192.168.1.73:8080"
export REMOTE_POLICY_DEVICE="cuda"
export REMOTE_ACTIONS_PER_CHUNK=25

POLICY_REPO="Zekai-Chen/pi05_fold_towel"
DATASET_REPO="Zekai-Chen/eval_bi_so101_rollout_4cam"
TASK="Fold the towel"
FPS=30
EPISODE_TIME=180
RESET_TIME=30
NUM_EPISODES=10
RESUME="true"

# First run: create dataset locally; subsequent runs: resume
if [ ! -d "$HOME/.cache/huggingface/lerobot/${DATASET_REPO}" ]; then
  RESUME="false"
fi

for arg in "$@"; do
  if [ "$arg" = "--fresh" ]; then
    echo "Starting fresh (clearing old data)..."
    rm -rf ~/.cache/huggingface/lerobot/${DATASET_REPO}
    RESUME="false"
  elif [[ "$arg" =~ ^[0-9]+$ ]]; then
    NUM_EPISODES=$arg
  fi
done

# Kill stale rerun window
pkill -f rerun 2>/dev/null
sleep 1

echo "=== Evo-RL Async Rollout with Intervention ==="
echo "Server:   ${REMOTE_POLICY_SERVER}"
echo "Policy:   ${POLICY_REPO}"
echo "Dataset:  ${DATASET_REPO}"
echo "Episodes: ${NUM_EPISODES}"
echo "Time/ep:  ${EPISODE_TIME}s (3 min)"
echo "Resume:   ${RESUME}"
echo "================================================"
echo "Press 'i' to intervene, 's' for success, 'f' for failure"

lerobot-human-inloop-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032350-if00 \
  --robot.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079934-if00 \
  --robot.id=bi_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1.1:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, top: {type: intelrealsense, serial_number_or_name: "827112073332", width: 640, height: 480, fps: 30, warmup_s: 10}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1.2:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "215322072684", width: 640, height: 480, fps: 30, warmup_s: 10}}' \
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
  --dataset.push_to_hub=false \
  --resume=${RESUME} \
  --display_data=true \
  --policy.path=${POLICY_REPO} \
  --policy.compile_model=false \
  --policy.device=cpu
