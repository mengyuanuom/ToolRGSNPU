#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG_PATH="${1:-config/realvlg/drogoff.yaml}"
CHECKPOINT_PATH="${2:?Usage: $0 [config] CHECKPOINT [GraspNet_VLG root]}"
DATA_ROOT="${3:-./datasets/GraspNet_VLG}"
NPU_ID="${NPU_ID:-0}"

for SPLIT_NAME in seen similar novel; do
  echo "[RealVLG] evaluating ${SPLIT_NAME}"
  python evaluate.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --npu "${NPU_ID}" --opts \
    DATA.root_path "${DATA_ROOT}" \
    TEST.test_split "${SPLIT_NAME}"
done
