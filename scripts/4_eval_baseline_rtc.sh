#!/bin/bash
# ============================================
# Step 4a: Evaluate baseline policy with RTC
# ============================================
# Usage:
#   bash scripts/4_eval_baseline_rtc.sh          # Run 180s (default)
#   bash scripts/4_eval_baseline_rtc.sh 300      # Run 300s
#
# RTC (Real-Time Chunking) smooths action chunk transitions.
# Press Ctrl+C to stop.

# --- Config ---
POLICY_REPO="Zekai-Chen/pi05_fold_towel"
TASK="Fold the towel"
FPS=30
DURATION=${1:-180}

echo "=== Evo-RL Baseline Evaluation (RTC) ==="
echo "Policy:   ${POLICY_REPO}"
echo "Duration: ${DURATION}s"
echo "FPS:      ${FPS}"
echo "========================================="

python examples/rtc/eval_with_real_robot.py \
  --policy.path=${POLICY_REPO} \
  --policy.device=cuda \
  --rtc.enabled=true \
  --rtc.execution_horizon=10 \
  --rtc.max_guidance_weight=10.0 \
  --rtc.prefix_attention_schedule=EXP \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032350-if00 \
  --robot.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079934-if00 \
  --robot.id=bi_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1.1:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, top: {type: intelrealsense, serial_number_or_name: "827112073332", width: 640, height: 480, fps: 30, warmup_s: 10}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:06:00.0-usb-0:1.2:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "215322072684", width: 640, height: 480, fps: 30, warmup_s: 10}}' \
  --task="${TASK}" \
  --duration=${DURATION} \
  --fps=${FPS}
