#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OCID_VLG_ROOT="${OCID_VLG_ROOT:-${REPO_ROOT}/datasets/OCID-VLG}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ocid_vlg_8models}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"

MODEL_NAMES=(
  crog
  lgd
  drog
  drogoff
  ggcnnclip
  grconvnetclip
  etrg_rgb
  maplegrasp
)
MODEL_CONFIGS=(
  config/ocid_vlg/crog.yaml
  config/ocid_vlg/lgd.yaml
  config/ocid_vlg/drog.yaml
  config/ocid_vlg/drogoff.yaml
  config/ocid_vlg/ggcnnclip.yaml
  config/ocid_vlg/grconvnetclip.yaml
  config/ocid_vlg/etrg.yaml
  config/ocid_vlg/maplegrasp.yaml
)

command -v python >/dev/null 2>&1 || {
  echo "[8models] python was not found in the active environment." >&2
  exit 127
}
[[ -d "${OCID_VLG_ROOT}" ]] || {
  echo "[8models] OCID-VLG dataset directory was not found: ${OCID_VLG_ROOT}" >&2
  exit 2
}

mkdir -p "${LOG_DIR}"
PID_FILE="${LOG_DIR}/${RUN_ID}.pids"
: >"${PID_FILE}"

launch_model() {
  local device="$1"
  local model_name="$2"
  local config_path="$3"
  local exp_name="${model_name}_ocid_vlg_npu${device}_${RUN_ID}"
  local log_file="${LOG_DIR}/${exp_name}.log"

  [[ -f "${config_path}" ]] || {
    echo "[8models] missing config: ${config_path}" >&2
    exit 2
  }

  ASCEND_RT_VISIBLE_DEVICES="${device}" \
    nohup python -u train.py \
      --config "${config_path}" \
      --npu 0 \
      --opts \
      DATA.root_path "${OCID_VLG_ROOT}" \
      TRAIN.exp_name "${exp_name}" \
      >"${log_file}" 2>&1 </dev/null &

  local pid=$!
  printf '%s %s npu=%s log=%s\n' \
    "${pid}" "${model_name}" "${device}" "${log_file}" >>"${PID_FILE}"
  echo "[8models] NPU ${device}: ${model_name} started, PID=${pid}"
  echo "[8models] log: ${log_file}"
}

for index in "${!MODEL_NAMES[@]}"; do
  launch_model \
    "${index}" \
    "${MODEL_NAMES[${index}]}" \
    "${MODEL_CONFIGS[${index}]}"
done

echo "[8models] all eight independent jobs were launched."
echo "[8models] batches come from YAML: ETRG=10/10, all other launched models=24/24."
echo "[8models] PID file: ${PID_FILE}"
