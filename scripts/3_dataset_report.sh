#!/bin/bash
# ============================================
# Step 3: Check dataset quality after collection
# ============================================
DATASET_REPO="${1:-deepreach/bi_so101_baseline_4cam}"

lerobot-dataset-report --dataset ${DATASET_REPO}
