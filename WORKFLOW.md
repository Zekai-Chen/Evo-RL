# Bimanual SO101 Towel Folding - Full Workflow Guide

## Project Info
- **Task**: Fold the towel (bimanual)
- **Hardware**: 2 SO101 leader arms + 2 SO101 follower arms, 2 USB wrist cameras, 2 D435 RealSense cameras
- **Dataset**: `Zekai-Chen/bi_so101_baseline_4cam` on HuggingFace
- **Camera views**: left_wrist, right_wrist, left_top, right_front

---

## Step 0: Setup on a New Machine

### 0.1 Clone and install
```bash
git clone https://github.com/Zekai-Chen/Evo-RL.git
cd Evo-RL
git checkout zekai-bimanual

conda create -y -n evo-rl python=3.10
conda activate evo-rl
pip install -e ".[feetech]"
pip install feetech-servo-sdk pyrealsense2
```

### 0.2 Copy calibration files
```bash
mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so_follower
mkdir -p ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader
cp calibration/robots/so_follower/*.json ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
cp calibration/teleoperators/so_leader/*.json ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
```

### 0.3 Login to HuggingFace
```bash
huggingface-cli login
# Enter your HF token (must have write access to Zekai-Chen org)
```

### 0.4 Login to GitHub (for pushing code)
```bash
gh auth login
# Select HTTPS, authenticate via browser
```

---

## Step 1: Hardware Check (MUST DO before collecting)

### 1.1 Find your serial ports (robot arms)
```bash
ls /dev/serial/by-id/
```
You need 4 ports. Match each to the correct arm by plugging one at a time:
- Left Leader: `usb-1a86_USB_Single_Serial_XXXXXXXX-if00`
- Right Leader: `usb-1a86_USB_Single_Serial_XXXXXXXX-if00`
- Left Follower: `usb-1a86_USB_Single_Serial_XXXXXXXX-if00`
- Right Follower: `usb-1a86_USB_Single_Serial_XXXXXXXX-if00`

**Known ports (OMEN laptop):**
| Arm | Serial ID |
|-----|-----------|
| Left Leader | `5B14110746` |
| Right Leader | `5B14032215` |
| Left Follower | `5B14032350` |
| Right Follower | `5AE6079934` |

### 1.2 Find your cameras
```bash
# USB cameras
ls /dev/v4l/by-path/

# RealSense cameras
python3 -c "import pyrealsense2 as rs; ctx=rs.context(); [print(d.get_info(rs.camera_info.serial_number), d.get_info(rs.camera_info.name)) for d in ctx.query_devices()]"
```

**Known cameras (OMEN laptop):**
| Camera | Type | ID |
|--------|------|----|
| Left Wrist | USB OpenCV | by-path (check `ls /dev/v4l/by-path/`) |
| Right Wrist | USB OpenCV | by-path (check `ls /dev/v4l/by-path/`) |
| Top D435 | RealSense | SN `827112073332` |
| Front D435 | RealSense | SN `215322072684` |

### 1.3 Check USB bus distribution
```bash
lsusb -t
```
**IMPORTANT**: Do NOT put all cameras on the same USB bus (480Mbps USB 2.0 will saturate). Spread cameras across different buses. Check with `lsusb -t` after plugging in.

### 1.4 Update scripts with your ports
Edit `scripts/1_teleop_test.sh` and `scripts/2_collect_success_data.sh` with the correct:
- Serial port paths for all 4 arms
- Camera by-path for both wrist cameras
- RealSense serial numbers for both D435s

### 1.5 Run teleop test
```bash
pkill -f rerun  # Kill any stale rerun windows
bash scripts/1_teleop_test.sh
```
Verify:
- All 4 arms respond to teleop
- All 4 camera feeds visible in rerun window
- Camera names are correct: left_wrist, right_wrist, left_top, right_front

---

## Step 2: Collect Success Data (target: 300 episodes)

### 2.1 Run the collection script
```bash
pkill -f rerun  # Always kill stale rerun first
bash scripts/2_collect_success_data.sh 10  # Collect 10 episodes per session
```
- Uses `--resume=true` by default, auto-downloads existing data from HuggingFace
- Auto-pushes to HuggingFace after each episode

### 2.2 Hotkeys during recording
| Key | Action |
|-----|--------|
| `s` | Mark episode as **SUCCESS** and save |
| `f` | Mark episode as **FAILURE** and save |
| Right Arrow | End current episode early |
| Left Arrow | End early and **re-record** current episode |
| Esc | Stop recording session |

**Rules:**
- Only press `s` when the towel is actually folded successfully
- Press `f` for failed attempts (these will be useful later for RL)
- For baseline data, aim for ALL success episodes
- If you forget to press `s`/`f`, the default is `failure`

### 2.3 If it crashes
1. `pkill -f rerun` (clear stale rerun window)
2. Check if the camera is still connected: `ls /dev/v4l/by-path/`
3. If camera missing, unplug and replug it
4. For RealSense timeout: unplug/replug the D435
5. Re-run the script — `--resume=true` will pick up where you left off
6. Data is auto-saved and auto-pushed, so no data loss on crash

### 2.4 Check data after collecting
```bash
# Quick report
bash scripts/3_dataset_report.sh

# Or manually check episode count
python3 -c "
import pandas as pd, glob, os
base = os.path.expanduser('~/.cache/huggingface/lerobot/Zekai-Chen/bi_so101_baseline_4cam/meta/episodes/chunk-000/')
files = sorted(glob.glob(os.path.join(base, '*.parquet')))
all_dfs = [pd.read_parquet(f, columns=['episode_index', 'episode_success']) for f in files]
combined = pd.concat(all_dfs)
print(f'Total episodes: {len(combined)}')
print(combined['episode_success'].value_counts())
"
```

### 2.5 Current progress
- **222 episodes collected** (all success, all pushed to HF)
- **Target: 300 episodes**
- **Remaining: ~78 episodes**

---

## Step 3: Train Baseline Policy (after 300 success episodes)

Train a baseline policy on the success-only data. This will have a LOW success rate — that's expected. RL will improve it.

```bash
lerobot-train \
  --dataset.repo_id=Zekai-Chen/bi_so101_baseline_4cam \
  --policy.type=pistar06 \
  --policy.dtype=bfloat16 \
  --batch_size=32 \
  --steps=30000 \
  --output_dir=outputs/train/baseline_towel_fold \
  --job_name=baseline_towel_fold \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=Zekai-Chen/bi_so101_baseline_policy
```

Multi-GPU (if available):
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --multi_gpu --num_processes=2 --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=Zekai-Chen/bi_so101_baseline_4cam \
  --policy.type=pistar06 \
  --policy.dtype=bfloat16 \
  --batch_size=16 \
  --steps=30000 \
  --output_dir=outputs/train/baseline_towel_fold \
  --job_name=baseline_towel_fold \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=Zekai-Chen/bi_so101_baseline_policy
```

---

## Step 4: Rollout Baseline Policy + Collect Mixed Data

Deploy the trained baseline policy with human-in-the-loop. The policy will attempt the task, you intervene when needed. This collects both success AND failure episodes.

```bash
lerobot-human-inloop-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=bi_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "<LEFT_WRIST_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, top: {type: intelrealsense, serial_number_or_name: "827112073332", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "<RIGHT_WRIST_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "215322072684", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=bi_leader \
  --dataset.repo_id=Zekai-Chen/bi_so101_rollout_round1 \
  --dataset.single_task="Fold the towel" \
  --dataset.num_episodes=100 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=30 \
  --dataset.push_to_hub=true \
  --display_data=true \
  --policy.path=outputs/train/baseline_towel_fold/checkpoints/last/pretrained_model \
  --resume=true
```

Hotkeys:
- `i` to toggle between policy control and manual intervention
- `s`/`f` to label success/failure
- Label HONESTLY — failures are valuable for RL training

---

## Step 5: Merge Datasets (optional)

If you collected rollout data in a separate dataset, merge it with the baseline data:

```bash
lerobot-edit-dataset \
  --repo_id=Zekai-Chen/bi_so101_merged_round1 \
  --operation.type=merge \
  --operation.repo_ids="['Zekai-Chen/bi_so101_baseline_4cam','Zekai-Chen/bi_so101_rollout_round1']"
```

Or just append to the same dataset by using `--dataset.repo_id=Zekai-Chen/bi_so101_baseline_4cam` in Step 4.

---

## Step 6: Train Value Function

Train the value function on the merged dataset (needs both success AND failure episodes):

```bash
lerobot-value-train \
  --dataset.repo_id=Zekai-Chen/bi_so101_merged_round1 \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.push_to_hub=true \
  --value.repo_id=Zekai-Chen/bi_so101_value_round1 \
  --batch_size=64 \
  --output_dir=outputs/value_train/round1 \
  --job_name=value_round1 \
  --wandb.enable=true
```

---

## Step 7: Value Inference (Compute Advantages)

Run value inference to compute advantage scores and binary indicators:

```bash
lerobot-value-infer \
  --dataset.repo_id=Zekai-Chen/bi_so101_merged_round1 \
  --inference.checkpoint_path=outputs/value_train/round1 \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_round1 \
  --acp.advantage_field=complementary_info.advantage_round1 \
  --acp.indicator_field=complementary_info.acp_indicator_round1 \
  --output_dir=outputs/value_infer/round1 \
  --job_name=value_round1.infer
```

This writes `value`, `advantage`, and `acp_indicator` columns back to the dataset.

---

## Step 8: Train ACP Policy (Advantage-Conditioned Policy)

Train the improved policy using advantage-conditioned tags:

```bash
lerobot-train \
  --dataset.repo_id=Zekai-Chen/bi_so101_merged_round1 \
  --policy.type=pistar06 \
  --policy.pretrained_path=outputs/train/baseline_towel_fold/checkpoints/last/pretrained_model \
  --policy.dtype=bfloat16 \
  --batch_size=32 \
  --steps=30000 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_round1 \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir=outputs/train/acp_round1 \
  --job_name=acp_round1 \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=Zekai-Chen/bi_so101_acp_policy_round1
```

---

## Step 9: Repeat (Next RL Round)

Go back to **Step 4** with the new ACP policy:
1. Rollout the ACP policy → collect more mixed data
2. Merge all data
3. Retrain value function
4. Rerun value inference
5. Retrain ACP policy

Each round should improve the success rate.

---

## Full Pipeline Summary

```
Step 2: Collect 300 success episodes (human teleop)
    |
Step 3: Train baseline policy (low success rate expected)
    |
Step 4: Rollout baseline + human intervention → mixed data
    |
Step 5: Merge baseline + rollout data
    |
Step 6: Train value function (on merged data)
    |
Step 7: Value inference → compute advantages
    |
Step 8: Train ACP policy (advantage-conditioned)
    |
Step 9: Repeat from Step 4 with improved policy
    └── Each round: higher success rate
```

---

## Troubleshooting

### Camera drops / USB errors
- Check `lsusb -t` to verify bus distribution
- **CRITICAL**: Do NOT put both wrist cameras on the same USB 2.0 bus (480Mbps). They WILL drop within seconds. Split them across different USB buses (one on USB 2.0, one on USB 3.0 via USB-A to USB-C adapter)
- D435 cameras MUST be on USB 3.0 (they cannot stream on USB 2.0 at all)
- Wrist camera `/dev/video*` numbers are unstable — they change after every disconnect/reconnect. Always check before running
- Unplug/replug cameras that time out
- `pkill -f rerun` before each recording session
- The `--resume=true` flag auto-continues from the last episode, so crashes don't lose data

### Motor communication errors
- Usually transient — just retry
- If persistent, check USB cable connection for the arm
- Verify with: `ls /dev/serial/by-id/`

### Data integrity check
```bash
python3 -c "
import pandas as pd, glob, os
base = os.path.expanduser('~/.cache/huggingface/lerobot/Zekai-Chen/bi_so101_baseline_4cam/meta/episodes/chunk-000/')
files = sorted(glob.glob(os.path.join(base, '*.parquet')))
all_dfs = [pd.read_parquet(f, columns=['episode_index', 'episode_success']) for f in files]
combined = pd.concat(all_dfs)
print(f'Total episodes: {len(combined)}')
print(combined['episode_success'].value_counts())
"
```

### Fresh start (delete local data and re-download from HF)
```bash
bash scripts/2_collect_success_data.sh 10 --fresh
```
