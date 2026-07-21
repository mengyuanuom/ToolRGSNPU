#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_START="${MASTER_PORT_START:-29500}"
SESSION_GAP_SECONDS="${SESSION_GAP_SECONDS:-3}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/grasp_tools_8npu}"

MODEL_NAMES=(crog drogoff lgd)
MODEL_CONFIGS=(
  config/grasp_tools/crog.yaml
  config/grasp_tools/drogoff_v2.yaml
  config/grasp_tools/lgd.yaml
)

ACTIVE_SESSION_PID=""

terminate_session() {
  local session_pid="${1:-}"
  [[ -n "${session_pid}" ]] || return 0

  if kill -0 "-${session_pid}" 2>/dev/null; then
    kill -TERM "-${session_pid}" 2>/dev/null || true
    for _ in {1..10}; do
      if ! kill -0 "-${session_pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi

  if kill -0 "-${session_pid}" 2>/dev/null; then
    kill -KILL "-${session_pid}" 2>/dev/null || true
  fi
  wait "${session_pid}" 2>/dev/null || true
}

handle_interrupt() {
  echo "[sequence] interruption received; stopping the active training session."
  terminate_session "${ACTIVE_SESSION_PID}"
  ACTIVE_SESSION_PID=""
  exit 130
}

trap handle_interrupt HUP INT TERM

run_model() {
  local model_name="$1"
  local config_path="$2"
  local master_port="$3"
  local timestamp
  local log_file
  local status

  timestamp="$(date '+%Y%m%d_%H%M%S')"
  log_file="${LOG_DIR}/${timestamp}_${model_name}.log"

  echo "[sequence] starting ${model_name} with ${NPROC_PER_NODE} NPUs"
  echo "[sequence] config: ${config_path}"
  echo "[sequence] log: ${log_file}"

  setsid bash -o pipefail -c '
    log_file="$1"
    shift
    "$@" 2>&1 | tee "${log_file}"
  ' toolrgs-training-session "${log_file}" \
    torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${master_port}" \
      train.py --config "${config_path}" &

  ACTIVE_SESSION_PID=$!
  set +e
  wait "${ACTIVE_SESSION_PID}"
  status=$?
  set -e

  # torchrun normally reaps all workers. This also closes any residual worker
  # process in the session before the next model reserves the NPU devices.
  terminate_session "${ACTIVE_SESSION_PID}"
  ACTIVE_SESSION_PID=""

  if [[ "${status}" -ne 0 ]]; then
    echo "[sequence] ${model_name} failed with exit code ${status}; stopping."
    return "${status}"
  fi

  echo "[sequence] ${model_name} completed; its training session is closed."
}

command -v torchrun >/dev/null 2>&1 || {
  echo "[sequence] torchrun was not found in the active environment." >&2
  exit 127
}
command -v setsid >/dev/null 2>&1 || {
  echo "[sequence] setsid was not found; install util-linux first." >&2
  exit 127
}

mkdir -p "${LOG_DIR}"

for index in "${!MODEL_NAMES[@]}"; do
  model_name="${MODEL_NAMES[${index}]}"
  config_path="${MODEL_CONFIGS[${index}]}"
  master_port=$((MASTER_PORT_START + index))

  [[ -f "${config_path}" ]] || {
    echo "[sequence] missing config: ${config_path}" >&2
    exit 2
  }

  run_model "${model_name}" "${config_path}" "${master_port}"

  if (( index + 1 < ${#MODEL_NAMES[@]} )); then
    echo "[sequence] waiting ${SESSION_GAP_SECONDS}s before the next fresh session."
    sleep "${SESSION_GAP_SECONDS}"
  fi
done

echo "[sequence] all three training jobs completed successfully."
