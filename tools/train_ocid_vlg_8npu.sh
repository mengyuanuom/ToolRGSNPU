#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# Preserve the launch terminal only for best-effort live display. Training and
# durable logs never depend on this descriptor remaining connected.
exec 3>&1

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_START="${MASTER_PORT_START:-29600}"
SESSION_GAP_SECONDS="${SESSION_GAP_SECONDS:-3}"
TAIL_FLUSH_SECONDS="${TAIL_FLUSH_SECONDS:-1}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/ocid_vlg_8npu}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
LIVE_OUTPUT="${LIVE_OUTPUT:-1}"
OCID_VLG_ROOT="${OCID_VLG_ROOT:-${REPO_ROOT}/datasets/OCID-VLG}"
START_FROM="${START_FROM:-}"

# CROGOFF is intentionally excluded. GraspMamba is also excluded because its
# upstream selective_scan_cuda operator has no supported Ascend implementation.
MODEL_NAMES=(
  crog
  drog
  drogoff
  maplegrasp
  ggcnnclip
  grconvnetclip
  lgd
  etrg_r50
  etrg_r101
)
MODEL_CONFIGS=(
  config/ocid_vlg/crog.yaml
  config/ocid_vlg/drog.yaml
  config/ocid_vlg/drogoff.yaml
  config/ocid_vlg/maplegrasp.yaml
  config/ocid_vlg/ggcnnclip.yaml
  config/ocid_vlg/grconvnetclip.yaml
  config/ocid_vlg/lgd.yaml
  config/ocid_vlg/etrg.yaml
  config/ocid_vlg/etrg_r101.yaml
)

ACTIVE_SESSION_PID=""
LOG_FOLLOWER_PID=""

live_output_enabled() {
  [[ "${LIVE_OUTPUT}" != "0" && "${LIVE_OUTPUT}" != "false" && "${LIVE_OUTPUT}" != "no" ]]
}

announce() {
  printf '%s\n' "$*"
  if live_output_enabled; then
    printf '%s\n' "$*" >&3 2>/dev/null || true
  fi
}

start_live_output() {
  local log_file="$1"
  LOG_FOLLOWER_PID=""
  live_output_enabled || return 0
  tail -n +1 -F "${log_file}" >&3 2>&3 &
  LOG_FOLLOWER_PID=$!
}

stop_live_output() {
  local follower_pid="${LOG_FOLLOWER_PID:-}"
  [[ -n "${follower_pid}" ]] || return 0
  if kill -0 "${follower_pid}" 2>/dev/null; then
    kill -TERM "${follower_pid}" 2>/dev/null || true
  fi
  wait "${follower_pid}" 2>/dev/null || true
  LOG_FOLLOWER_PID=""
}

stop_active_training() {
  local torchrun_pid="${ACTIVE_SESSION_PID:-}"
  [[ -n "${torchrun_pid}" ]] || return 0
  if kill -0 "${torchrun_pid}" 2>/dev/null; then
    kill -TERM "${torchrun_pid}" 2>/dev/null || true
  fi
  wait "${torchrun_pid}" 2>/dev/null || true
}

handle_interrupt() {
  local signal_name="$1"
  announce "[sequence] received ${signal_name}; stopping the active torchrun agent."
  stop_live_output
  stop_active_training
  ACTIVE_SESSION_PID=""
  exit 130
}

# Keep a direct invocation alive when its SSH pseudo-terminal disconnects.
trap '' HUP
trap 'handle_interrupt SIGINT' INT
trap 'handle_interrupt SIGTERM' TERM

run_model() {
  local model_name="$1"
  local config_path="$2"
  local master_port="$3"
  local timestamp
  local log_file
  local exp_name
  local status

  timestamp="$(date '+%Y%m%d_%H%M%S')"
  log_file="${LOG_DIR}/${RUN_ID}_${timestamp}_${model_name}.log"
  exp_name="${model_name}_ocid_vlg_8npu_${RUN_ID}"

  announce "[sequence] starting ${model_name} with ${NPROC_PER_NODE} NPUs"
  announce "[sequence] config: ${config_path}"
  announce "[sequence] dataset: ${OCID_VLG_ROOT}"
  announce "[sequence] experiment: ${exp_name}"
  announce "[sequence] log: ${log_file}"

  torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${master_port}" \
      train.py --config "${config_path}" --opts \
      DATA.root_path "${OCID_VLG_ROOT}" \
      TRAIN.exp_name "${exp_name}" \
      >>"${log_file}" 2>&1 </dev/null &

  ACTIVE_SESSION_PID=$!
  start_live_output "${log_file}"
  set +e
  wait "${ACTIVE_SESSION_PID}"
  status=$?
  set -e

  sleep "${TAIL_FLUSH_SECONDS}"
  stop_live_output
  ACTIVE_SESSION_PID=""

  if [[ "${status}" -ne 0 ]]; then
    announce "[sequence] ${model_name} failed with exit code ${status}; stopping."
    return "${status}"
  fi

  announce "[sequence] ${model_name} completed; its training session is closed."
}

command -v torchrun >/dev/null 2>&1 || {
  echo "[sequence] torchrun was not found in the active environment." >&2
  exit 127
}
[[ -d "${OCID_VLG_ROOT}" ]] || {
  echo "[sequence] OCID-VLG dataset directory was not found: ${OCID_VLG_ROOT}" >&2
  echo "[sequence] set OCID_VLG_ROOT=/absolute/path/to/OCID-VLG and retry." >&2
  exit 2
}
[[ -f "${OCID_VLG_ROOT}/refer/multiple/train_expressions.json" ]] || {
  echo "[sequence] missing OCID-VLG train expressions under: ${OCID_VLG_ROOT}" >&2
  exit 2
}

mkdir -p "${LOG_DIR}"
SEQUENCE_LOG="${LOG_DIR}/${RUN_ID}_sequence.log"
echo "[sequence] durable log: ${SEQUENCE_LOG}"
exec </dev/null >>"${SEQUENCE_LOG}" 2>&1
announce "[sequence] run id: ${RUN_ID}"
announce "[sequence] CROGOFF and unsupported GraspMamba are excluded."
announce "[sequence] ${#MODEL_NAMES[@]} OCID-VLG models are scheduled."

start_index=0
if [[ -n "${START_FROM}" ]]; then
  found_start=0
  for index in "${!MODEL_NAMES[@]}"; do
    if [[ "${MODEL_NAMES[${index}]}" == "${START_FROM}" ]]; then
      start_index="${index}"
      found_start=1
      break
    fi
  done
  if [[ "${found_start}" -ne 1 ]]; then
    echo "[sequence] unknown START_FROM model: ${START_FROM}" >&2
    echo "[sequence] available: ${MODEL_NAMES[*]}" >&2
    exit 2
  fi
  announce "[sequence] resuming sequence from model: ${START_FROM}"
fi

for index in "${!MODEL_NAMES[@]}"; do
  (( index < start_index )) && continue
  model_name="${MODEL_NAMES[${index}]}"
  config_path="${MODEL_CONFIGS[${index}]}"
  master_port=$((MASTER_PORT_START + index))

  [[ -f "${config_path}" ]] || {
    echo "[sequence] missing config: ${config_path}" >&2
    exit 2
  }

  run_model "${model_name}" "${config_path}" "${master_port}"

  if (( index + 1 < ${#MODEL_NAMES[@]} )); then
    announce "[sequence] waiting ${SESSION_GAP_SECONDS}s before the next fresh session."
    sleep "${SESSION_GAP_SECONDS}"
  fi
done

announce "[sequence] all scheduled OCID-VLG training jobs completed successfully."
