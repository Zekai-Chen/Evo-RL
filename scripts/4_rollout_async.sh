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
SERVER="192.168.1.73:8080"
POLICY_REPO="Zekai-Chen/pi05_fold_towel"
DATASET_REPO="Zekai-Chen/eval_bi_so101_rollout_4cam"
NUM_EPISODES=${1:-50}
FRESH_FLAG=""

for arg in "$@"; do
  if [ "$arg" = "--fresh" ]; then
    FRESH_FLAG="--fresh"
  fi
done

# Kill stale rerun window
pkill -f rerun 2>/dev/null
sleep 1

echo "=== Evo-RL Async Rollout with Intervention ==="
echo "Server:   ${SERVER}"
echo "Policy:   ${POLICY_REPO}"
echo "Dataset:  ${DATASET_REPO}"
echo "Episodes: ${NUM_EPISODES}"
echo "================================================"
echo "Press 'i' to intervene, 's' for success, 'f' for failure"

python examples/rollout_async.py \
  --server=${SERVER} \
  --policy_type=pi05 \
  --policy_path=${POLICY_REPO} \
  --policy_device=cuda \
  --num_episodes=${NUM_EPISODES} \
  --dataset_repo=${DATASET_REPO} \
  ${FRESH_FLAG}
