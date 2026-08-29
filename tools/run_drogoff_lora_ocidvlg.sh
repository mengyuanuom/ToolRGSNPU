#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="/data1/ma00959358/pangu/ToolRGSNPU"
ENV_SCRIPT="/data1/wangxuefei/Ascend/ascend-toolkit/set_env.sh"
TORCHRUN="/root/miniconda3/envs/pangu_mmy/bin/torchrun"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
CONFIG="config/ocid_vlg/drogoff_lora.yaml"
MODE="${1:-}"

case "${MODE}" in
  smoke)
    EXP_NAME="drogoff_lora_r24_a48_12l_ocid_vlg_smoke_e1_8npu_bs128"
    LOG_NAME="launcher.smoke.log"
    EXTRA_OPTS=(TRAIN.epochs 1 TRAIN.evaluate False TRAIN.print_freq 1)
    ;;
  train)
    EXP_NAME="drogoff_lora_r24_a48_12l_ocid_vlg_full_e50_8npu_bs128"
    LOG_NAME="launcher.train.log"
    EXTRA_OPTS=()
    ;;
  *)
    echo "usage: $0 {smoke|train}" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"
if pgrep -f "/torchrun" >/dev/null; then
  echo "refusing to start: an existing torchrun is active" >&2
  pgrep -af "/torchrun" >&2
  exit 9
fi

NPU_IDLE_COUNT="$(npu-smi info 2>/dev/null | grep -c "No running processes found" || true)"
if [[ "${NPU_IDLE_COUNT}" -ne 8 ]]; then
  echo "refusing to start: expected 8 idle NPUs, found ${NPU_IDLE_COUNT}" >&2
  npu-smi info >&2
  exit 9
fi

OUTPUT_DIR="exp/ocid_vlg/${EXP_NAME}"
LOG_PATH="${OUTPUT_DIR}/${LOG_NAME}"
mkdir -p "${OUTPUT_DIR}"
if [[ -e "${LOG_PATH}" ]]; then
  echo "refusing to overwrite existing log: ${LOG_PATH}" >&2
  exit 10
fi

source "${ENV_SCRIPT}"
nohup "${TORCHRUN}" --standalone --nproc_per_node=8   train.py --config "${CONFIG}" --opts   TRAIN.exp_name "${EXP_NAME}" "${EXTRA_OPTS[@]}"   >"${LOG_PATH}" 2>&1 </dev/null &
LAUNCHER_PID=$!

sleep 3
if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
  echo "launcher exited during startup; inspect ${LOG_PATH}" >&2
  tail -n 80 "${LOG_PATH}" >&2 || true
  exit 11
fi

echo "mode=${MODE}"
echo "launcher_pid=${LAUNCHER_PID}"
echo "log=${LOG_PATH}"
