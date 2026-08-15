#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VCOT_ROOT="${VCOT_ROOT:-${REPO_ROOT}/datasets/graspanything-vcot}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/vcot_8models_queue}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
POLL_SECONDS="${POLL_SECONDS:-20}"

# The first four entries are deliberately pinned to the first launch wave.
MODEL_NAMES=(
  drog
  drogoff
  crog
  crogoff
  ggcnnclip
  grconvnetclip
  maplegrasp
  lgd
)
MODEL_CONFIGS=(
  config/vcot/drog.yaml
  config/vcot/drogoff.yaml
  config/vcot/crog.yaml
  config/vcot/crogoff.yaml
  config/vcot/ggcnnclip.yaml
  config/vcot/grconvnetclip.yaml
  config/vcot/maplegrasp.yaml
  config/vcot/lgd.yaml
)

command -v python >/dev/null 2>&1 || {
  echo "[vcot-queue] python was not found in the active environment." >&2
  exit 127
}
[[ -d "${VCOT_ROOT}" ]] || {
  echo "[vcot-queue] VCoT dataset directory was not found: ${VCOT_ROOT}" >&2
  exit 2
}
[[ -d "${VCOT_ROOT}/split/vcot" ]] || {
  echo "[vcot-queue] VCoT split directory was not found: ${VCOT_ROOT}/split/vcot" >&2
  exit 2
}

for config_path in "${MODEL_CONFIGS[@]}"; do
  [[ -f "${config_path}" ]] || {
    echo "[vcot-queue] missing config: ${config_path}" >&2
    exit 2
  }
done

mkdir -p "${LOG_DIR}"
PID_FILE="${LOG_DIR}/${RUN_ID}.pids"
STATUS_FILE="${LOG_DIR}/${RUN_ID}.status.tsv"
: >"${PID_FILE}"
printf 'model\tnpu\tpid\tstatus\tlog\n' >"${STATUS_FILE}"
printf '%s scheduler npu=all\n' "$$" >>"${PID_FILE}"

declare -A ACTIVE_PIDS=()
declare -A ACTIVE_NAMES=()
declare -A ACTIVE_LOGS=()
declare -A ACTIVE_STARTS=()
FAILED_JOBS=0

launch_model() {
  local device="$1"
  local model_index="$2"
  local model_name="${MODEL_NAMES[${model_index}]}"
  local config_path="${MODEL_CONFIGS[${model_index}]}"
  local exp_name="${model_name}_vcot_npu${device}_${RUN_ID}"
  local log_file="${LOG_DIR}/${exp_name}.log"

  ASCEND_RT_VISIBLE_DEVICES="${device}" \
    python -u train.py \
      --config "${config_path}" \
      --npu 0 \
      --opts \
      DATA.root_path "${VCOT_ROOT}" \
      DATA.split_root "${VCOT_ROOT}/split/vcot" \
      TRAIN.batch_size "${PER_DEVICE_BATCH_SIZE}" \
      TRAIN.exp_name "${exp_name}" \
      >"${log_file}" 2>&1 </dev/null &

  local pid=$!
  ACTIVE_PIDS[${device}]="${pid}"
  ACTIVE_NAMES[${device}]="${model_name}"
  ACTIVE_LOGS[${device}]="${log_file}"
  ACTIVE_STARTS[${device}]="$(date '+%F %T')"
  printf '%s %s npu=%s log=%s\n' \
    "${pid}" "${model_name}" "${device}" "${log_file}" >>"${PID_FILE}"
  echo "[vcot-queue] NPU ${device}: ${model_name} started, PID=${pid}, batch=${PER_DEVICE_BATCH_SIZE}"
  echo "[vcot-queue] log: ${log_file}"
}

stop_children() {
  trap - INT TERM
  echo "[vcot-queue] stopping active training children..."
  for device in "${!ACTIVE_PIDS[@]}"; do
    local pid="${ACTIVE_PIDS[${device}]:-}"
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap stop_children INT TERM

next_index=0
for device in {0..7}; do
  if (( next_index < ${#MODEL_NAMES[@]} )); then
    launch_model "${device}" "${next_index}"
    ((next_index += 1))
  fi
done

while :; do
  active_count=0
  for device in {0..7}; do
    pid="${ACTIVE_PIDS[${device}]:-}"
    [[ -n "${pid}" ]] || continue
    ((active_count += 1))

    if kill -0 "${pid}" 2>/dev/null; then
      continue
    fi

    if wait "${pid}"; then
      status=0
    else
      status=$?
      ((FAILED_JOBS += 1))
    fi
    model_name="${ACTIVE_NAMES[${device}]}"
    log_file="${ACTIVE_LOGS[${device}]}"
    echo "[vcot-queue] NPU ${device}: ${model_name} finished with status ${status}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${model_name}" "${device}" "${pid}" "${status}" "${log_file}" >>"${STATUS_FILE}"
    unset 'ACTIVE_PIDS['"${device}"']' 'ACTIVE_NAMES['"${device}"']' \
      'ACTIVE_LOGS['"${device}"']' 'ACTIVE_STARTS['"${device}"']'
    ((active_count -= 1))

    if (( next_index < ${#MODEL_NAMES[@]} )); then
      launch_model "${device}" "${next_index}"
      ((next_index += 1))
      ((active_count += 1))
    fi
  done

  if (( active_count == 0 && next_index >= ${#MODEL_NAMES[@]} )); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "[vcot-queue] all ${#MODEL_NAMES[@]} jobs finished; failures=${FAILED_JOBS}"
echo "[vcot-queue] status file: ${STATUS_FILE}"
(( FAILED_JOBS == 0 ))
