#!/usr/bin/env bash
set -Eeo pipefail

PROJECT=/data1/ma00959358/pangu/ToolRGSNPU
TORCHRUN=/root/miniconda3/envs/pangu_mmy/bin/torchrun
PYTHON=/root/miniconda3/envs/pangu_mmy/bin/python
ASCEND_ENV=/data1/wangxuefei/Ascend/ascend-toolkit/set_env.sh
PROFILE="${1:-}"
NPU_POLL_SECONDS="${NPU_POLL_SECONDS:-10}"

usage() {
  echo "usage: $0 {baselines_bs256|drogoff_transport}" >&2
}

case "${PROFILE}" in
  baselines_bs256)
    STAGE_NAMES=(ocid_grconvnetclip vcot_maplegrasp vcot_etrg)
    STAGE_CONFIGS=(
      config/experiments/ocid_vlg/grconvnetclip_bs256.yaml
      config/experiments/vcot/maplegrasp_bs256.yaml
      config/experiments/vcot/etrg_bs256.yaml
    )
    STAGE_DIRS=(
      exp/ocid_vlg/grconvnetclip_ocid_vlg_8npu_bs256_e50
      exp/vcot/maplegrasp_vcot_8npu_bs256_e36
      exp/vcot/etrg_vcot_8npu_bs256_e36
    )
    ;;
  drogoff_transport)
    STAGE_NAMES=(realvlg ocid_vlg vcot)
    STAGE_CONFIGS=(
      config/experiments/realvlg/drogoff_transport.yaml
      config/experiments/ocid_vlg/drogoff_transport.yaml
      config/experiments/vcot/drogoff_transport.yaml
    )
    STAGE_DIRS=(
      exp/realvlg/drogoff_transport_realvlg_full_e24_8npu_bs256
      exp/ocid_vlg/drogoff_transport_ocid_vlg_8npu
      exp/vcot/drogoff_transport_vcot_8npu_bs128_e36
    )
    ;;
  *)
    usage
    exit 2
    ;;
esac

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
cd "${PROJECT}"
# Ascend's setup script reads optional variables before assigning defaults.
# Load it before enabling nounset for the queue itself.
# shellcheck disable=SC1090
source "${ASCEND_ENV}"
set -u
# shellcheck disable=SC1091
source "${PROJECT}/tools/lib/torchrun_process_guard.sh"
torchrun_guard_install

wait_for_all_npus() {
  while true; do
    if pgrep -af '[t]orchrun' >/dev/null || pgrep -af '[t]rain.py' >/dev/null; then
      sleep "${NPU_POLL_SECONDS}"
      continue
    fi
    local free_npus
    free_npus="$(npu-smi info 2>/dev/null | grep -c 'No running processes found in NPU' || true)"
    if [[ "${free_npus}" -eq 8 ]]; then
      return 0
    fi
    sleep "${NPU_POLL_SECONDS}"
  done
}

next_log() {
  local exp_dir="$1"
  local candidate="${exp_dir}/launcher.train.log"
  if [[ -e "${candidate}" ]]; then
    candidate="${exp_dir}/launcher.train.$(date +%Y%m%d_%H%M%S).log"
  fi
  printf '%s\n' "${candidate}"
}

verify_checkpoint() {
  local checkpoint="$1"
  "${PYTHON}" -c \
    'import sys, torch; p=sys.argv[1]; c=torch.load(p, map_location="cpu"); assert isinstance(c, dict); assert "state_dict" in c; print("CHECKPOINT_OK", p, "epoch", c.get("epoch"))' \
    "${checkpoint}"
}

run_stage() {
  local name="$1"
  local config="$2"
  local exp_dir="$3"

  wait_for_all_npus
  if [[ -e "${exp_dir}/last.pth" ]]; then
    echo "REFUSE_EXISTING_CHECKPOINT ${name} ${exp_dir}/last.pth"
    return 20
  fi

  mkdir -p "${exp_dir}"
  local launcher_log
  launcher_log="$(next_log "${exp_dir}")"
  echo "STAGE_START ${name} $(date --iso-8601=seconds) config=${config} log=${launcher_log}"

  torchrun_guard_start \
    "${launcher_log}" \
    "${TORCHRUN}" --standalone --nproc_per_node=8 \
    train.py --config "${config}"

  local rc=0
  torchrun_guard_wait || rc=$?
  echo "STAGE_END ${name} $(date --iso-8601=seconds) rc=${rc} log=${launcher_log}"
  if [[ "${rc}" -ne 0 ]]; then
    return "${rc}"
  fi
  if ! grep -q 'Training time' "${launcher_log}"; then
    echo "MISSING_TRAINING_TIME ${name} ${launcher_log}"
    return 21
  fi
  if grep -Eqi \
      'out of memory|(^|[^[:alpha:]])nan([^[:alpha:]]|$)|hccl.*(error|fail)|traceback|state_dict.*(error|mismatch)' \
      "${launcher_log}"; then
    echo "HARD_ERROR_PATTERN ${name} ${launcher_log}"
    return 22
  fi
  verify_checkpoint "${exp_dir}/last.pth"
}

for index in "${!STAGE_NAMES[@]}"; do
  run_stage \
    "${STAGE_NAMES[${index}]}" \
    "${STAGE_CONFIGS[${index}]}" \
    "${STAGE_DIRS[${index}]}"
done

echo "ALL_STAGES_COMPLETE profile=${PROFILE} $(date --iso-8601=seconds)"
