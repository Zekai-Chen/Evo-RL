# Bimanual SO101 Async Inference + Rollout Guide

This guide covers how to run pi05 policy inference on a remote GPU server and collect rollout data with human intervention on the local robot controller.

## Architecture

```
[OMEN Laptop (RTX 5060 Ti)]          [Workstation (RTX 5090)]
  - Robot control (bi_so101)            - Policy inference (pi05)
  - 4 cameras (2 wrist + 2 D435)       - gRPC PolicyServer on :8080
  - Teleop (leader arms)               - Preprocessing + Postprocessing
  - Data recording (LeRobotDataset)
  - Keyboard events (i/s/f)
       ↕ gRPC (LAN) ↕
```

## Prerequisites

### Local machine (robot controller)
```bash
conda activate evo-rl
pip install -e ".[async]"
# PyTorch nightly for RTX 5060 Ti (sm_120):
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
# Custom transformers for pi05:
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
```

### Remote server (inference)
```bash
conda activate evo-rl
pip install -e ".[async,pi]"
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
# Accept gated model access: https://huggingface.co/google/paligemma-3b-pt-224
huggingface-cli login
```

Sync code from fork:
```bash
git remote add fork https://github.com/Zekai-Chen/Evo-RL.git
git fetch fork
git checkout fork/zekai-bimanual -- src/lerobot/async_inference/configs.py \
  src/lerobot/async_inference/policy_server.py \
  src/lerobot/policies/pi05/modeling_pi05.py
```

## Step 1: Start Policy Server (remote)

```bash
TORCH_COMPILE_DISABLE=1 python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080
```

Wait for `PolicyServer started on 0.0.0.0:8080`.

To add RTC (Real-Time Chunking) for smoother action transitions:
```bash
TORCH_COMPILE_DISABLE=1 python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --rtc_config.enabled=true \
  --rtc_config.execution_horizon=10 \
  --rtc_config.max_guidance_weight=10.0 \
  --rtc_config.prefix_attention_schedule=EXP
```

## Step 2: Run Evaluation (no recording)

Pure async inference to observe baseline performance:
```bash
bash scripts/4_eval_baseline_async.sh
```

## Step 3: Run Rollout with Intervention (recording)

Async inference + recording + human intervention:
```bash
bash scripts/4_rollout_async.sh --fresh     # First run
bash scripts/4_rollout_async.sh             # Resume
bash scripts/4_rollout_async.sh 100         # 100 episodes
```

### Hotkeys
| Key | Action |
|-----|--------|
| `i` | Toggle intervention (policy ↔ human teleop) |
| `s` | Mark episode as SUCCESS, end episode |
| `f` | Mark episode as FAILURE, end episode |
| `→` | End current episode early |
| `Esc` | Stop recording session |

### How it works
- `REMOTE_POLICY_SERVER` env var triggers `RemotePolicy` creation in `make_policy()`
- No model loaded locally — all inference on the remote server
- Raw robot observations sent via gRPC, postprocessed actions returned
- Recording, intervention, and episode labeling work normally

## Hardware Setup

### USB Layout (10Gbps hub)
| Port | Device |
|------|--------|
| 0:1.1 | Left wrist camera (USB2.0_CAM1) |
| 0:1.2 | Right wrist camera (USB2.0_CAM1) |
| 0:1.3 | Top D435 (827112073332) |
| 0:1.4 | Front D435 (215322072684) |

### Serial Ports (stable by-id)
| Device | Serial |
|--------|--------|
| Left Follower | 5B14032350 |
| Right Follower | 5AE6079934 |
| Left Leader | 5B14110746 |
| Right Leader | 5B14032215 |

## Troubleshooting

### Server CUDA OOM
Each client reconnection loads a new model copy. Restart server between sessions:
```bash
# Ctrl+C server, then restart
TORCH_COMPILE_DISABLE=1 python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080
```

### Servo communication errors
Random `Failed to write 'Torque_Enable'` errors: unplug/replug the affected arm's USB serial cable, wait 3 seconds, retry.

### Camera not detected
Check USB devices:
```bash
ls /dev/v4l/by-path/ | grep "usb-0:1\.[12]"
```
If missing, unplug/replug the camera. If persistent, try a different USB port.

### transformers version
pi05 requires the custom transformers branch:
```bash
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
python -c "from transformers.models.siglip import check; print(check.check_whether_transformers_replace_is_installed_correctly())"
# Should print: True
```

### PyTorch version reset
If torch gets downgraded (e.g., by pip installing other packages):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/1_teleop_test.sh` | Test teleop + 4 cameras |
| `scripts/2_collect_success_data.sh` | Collect success demonstrations |
| `scripts/4_eval_baseline.sh` | Local eval (needs CUDA) |
| `scripts/4_eval_baseline_async.sh` | Async eval (remote inference) |
| `scripts/4_eval_baseline_rtc.sh` | Local RTC eval (needs CUDA) |
| `scripts/4_rollout_with_intervention.sh` | Local rollout (needs CUDA) |
| `scripts/4_rollout_async.sh` | Async rollout with intervention |
