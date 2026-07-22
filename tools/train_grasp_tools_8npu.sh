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
MASTER_PORT_START="${MASTER_PORT_START:-29500}"
SESSION_GAP_SECONDS="${SESSION_GAP_SECONDS:-3}"
TAIL_FLUSH_SECONDS="${TAIL_FLUSH_SECONDS:-1}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/grasp_tools_8npu}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
LIVE_OUTPUT="${LIVE_OUTPUT:-1}"

MODEL_NAMES=(crog drogoff lgd)
MODEL_CONFIGS=(
  config/grasp_tools/crog.yaml
  config/grasp_tools/drogoff_v2.yaml
  config/grasp_tools/lgd.yaml
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
    # Signal only the torchrun agent. It owns orderly worker shutdown; the
    # sequence script never sends process-group signals during normal flow.
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

# SSH disconnects deliver SIGHUP. Keep the sequence and its current torchrun
# session alive; only explicit interrupt/termination signals stop the workers.
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
  exp_name="${model_name}_grasp_tools_v2_8npu_${RUN_ID}"

  announce "[sequence] starting ${model_name} with ${NPROC_PER_NODE} NPUs"
  announce "[sequence] config: ${config_path}"
  announce "[sequence] experiment: ${exp_name}"
  announce "[sequence] log: ${log_file}"

  torchrun \
      --nnodes=1 \
      --node_rank=0 \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${master_port}" \
      train.py --config "${config_path}" --opts \
      TRAIN.exp_name "${exp_name}" \
      >>"${log_file}" 2>&1 </dev/null &

  ACTIVE_SESSION_PID=$!
  start_live_output "${log_file}"
  set +e
  wait "${ACTIVE_SESSION_PID}"
  status=$?
  set -e

  # Give tail one final polling window to mirror the last buffered lines. The
  # complete output is already durable in log_file even if the terminal left.
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
mkdir -p "${LOG_DIR}"
SEQUENCE_LOG="${LOG_DIR}/${RUN_ID}_sequence.log"
echo "[sequence] durable log: ${SEQUENCE_LOG}"
# Do not leave stdout/stderr attached to an SSH pseudo-terminal. This makes a
# direct invocation survive terminal disconnects without requiring tmux/nohup.
exec </dev/null >>"${SEQUENCE_LOG}" 2>&1
announce "[sequence] run id: ${RUN_ID}"

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
    announce "[sequence] waiting ${SESSION_GAP_SECONDS}s before the next fresh session."
    sleep "${SESSION_GAP_SECONDS}"
  fi
done

announce "[sequence] all three training jobs completed successfully."
