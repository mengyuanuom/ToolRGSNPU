#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_START="${MASTER_PORT_START:-29700}"
SESSION_GAP_SECONDS="${SESSION_GAP_SECONDS:-5}"
VCOT_ROOT="${VCOT_ROOT:-${REPO_ROOT}/datasets/graspanything-vcot}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/vcot_8npu_sequence}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
START_FROM="${START_FROM:-}"

# Priority order requested for the VCoT comparison. The repeated CROG in the
# request is interpreted as DROG, matching the four previously named models.
MODEL_NAMES=(drogoff crog drog crogoff)
MODEL_CONFIGS=(
  config/vcot/drogoff.yaml
  config/vcot/crog.yaml
  config/vcot/drog.yaml
  config/vcot/crogoff.yaml
)

ACTIVE_TORCHRUN_PID=""

stop_active_training() {
  local pid="${ACTIVE_TORCHRUN_PID:-}"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

handle_interrupt() {
  echo "[vcot-sequence] stopping active torchrun session"
  stop_active_training
  ACTIVE_TORCHRUN_PID=""
  exit 130
}

trap '' HUP
trap handle_interrupt INT TERM

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
  exp_name="${model_name}_vcot_8npu_b${GLOBAL_BATCH_SIZE}_${RUN_ID}"

  echo "[vcot-sequence] starting ${model_name}"
  echo "[vcot-sequence] config=${config_path}"
  echo "[vcot-sequence] experiment=${exp_name}"
  echo "[vcot-sequence] log=${log_file}"

  torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${master_port}" \
    train.py --config "${config_path}" --opts \
    DATA.root_path "${VCOT_ROOT}" \
    DATA.split_root "${VCOT_ROOT}/split/vcot" \
    TRAIN.batch_size "${GLOBAL_BATCH_SIZE}" \
    TRAIN.exp_name "${exp_name}" \
    >>"${log_file}" 2>&1 </dev/null &

  ACTIVE_TORCHRUN_PID=$!
  echo "${ACTIVE_TORCHRUN_PID} ${model_name} ${log_file}" >>"${PID_FILE}"
  set +e
  wait "${ACTIVE_TORCHRUN_PID}"
  status=$?
  set -e
  ACTIVE_TORCHRUN_PID=""

  if [[ "${status}" -ne 0 ]]; then
    echo "[vcot-sequence] ${model_name} failed with exit code ${status}; sequence stopped"
    return "${status}"
  fi
  echo "[vcot-sequence] ${model_name} completed"
}

command -v torchrun >/dev/null 2>&1 || {
  echo "[vcot-sequence] torchrun was not found in the active environment" >&2
  exit 127
}
[[ "${NPROC_PER_NODE}" -eq 8 ]] || {
  echo "[vcot-sequence] this schedule requires NPROC_PER_NODE=8" >&2
  exit 2
}
[[ "$((GLOBAL_BATCH_SIZE % NPROC_PER_NODE))" -eq 0 ]] || {
  echo "[vcot-sequence] global batch must be divisible by NPROC_PER_NODE" >&2
  exit 2
}
[[ -d "${VCOT_ROOT}" && -d "${VCOT_ROOT}/split/vcot" ]] || {
  echo "[vcot-sequence] VCoT dataset or split directory is missing under ${VCOT_ROOT}" >&2
  exit 2
}

mkdir -p "${LOG_DIR}"
SEQUENCE_LOG="${LOG_DIR}/${RUN_ID}_sequence.log"
PID_FILE="${LOG_DIR}/${RUN_ID}.pids"
: >"${PID_FILE}"
exec </dev/null >>"${SEQUENCE_LOG}" 2>&1
echo "[vcot-sequence] run_id=${RUN_ID}"
echo "[vcot-sequence] models=${MODEL_NAMES[*]}"
echo "[vcot-sequence] global_batch=${GLOBAL_BATCH_SIZE}, per_rank_batch=$((GLOBAL_BATCH_SIZE / NPROC_PER_NODE))"

start_index=0
if [[ -n "${START_FROM}" ]]; then
  found=0
  for index in "${!MODEL_NAMES[@]}"; do
    if [[ "${MODEL_NAMES[${index}]}" == "${START_FROM}" ]]; then
      start_index="${index}"
      found=1
      break
    fi
  done
  [[ "${found}" -eq 1 ]] || {
    echo "[vcot-sequence] unknown START_FROM=${START_FROM}" >&2
    exit 2
  }
fi

for index in "${!MODEL_NAMES[@]}"; do
  (( index < start_index )) && continue
  config_path="${MODEL_CONFIGS[${index}]}"
  [[ -f "${config_path}" ]] || {
    echo "[vcot-sequence] missing config: ${config_path}" >&2
    exit 2
  }
  run_model "${MODEL_NAMES[${index}]}" "${config_path}" "$((MASTER_PORT_START + index))"
  if (( index + 1 < ${#MODEL_NAMES[@]} )); then
    sleep "${SESSION_GAP_SECONDS}"
  fi
done

echo "[vcot-sequence] all requested models completed successfully"
