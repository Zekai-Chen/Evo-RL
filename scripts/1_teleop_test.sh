#!/bin/bash
# ============================================
# Step 1: Verify bimanual teleop + 4 cameras
# ============================================
# Run this first to confirm all arms and cameras work.
# You should see 4 camera feeds (left_wrist, right_wrist, top, side).
# Press Ctrl+C to stop.

lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032350-if00 \
  --robot.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6079934-if00 \
  --robot.id=bi_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:09:00.3-usb-0:1:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, top: {type: intelrealsense, serial_number_or_name: "827112073332", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/pci-0000:09:00.3-usb-0:2:1.0-video-index0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "215322072684", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14110746-if00 \
  --teleop.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032215-if00 \
  --teleop.id=bi_leader \
  --display_data=true
