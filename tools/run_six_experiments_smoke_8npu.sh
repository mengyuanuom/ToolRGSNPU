#!/usr/bin/env bash
set -Eeo pipefail

PROJECT=/data1/ma00959358/pangu/ToolRGSNPU
TORCHRUN=/root/miniconda3/envs/pangu_mmy/bin/torchrun
PYTHON=/root/miniconda3/envs/pangu_mmy/bin/python
ASCEND_ENV=/data1/wangxuefei/Ascend/ascend-toolkit/set_env.sh
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
POLL_SECONDS="${POLL_SECONDS:-5}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
cd "${PROJECT}"
source "${ASCEND_ENV}"
set -u
source "${PROJECT}/tools/lib/torchrun_process_guard.sh"
torchrun_guard_install

NAMES=(graspmamba_ocid_vlg graspmamba_vcot graspmamba_realvlg maplegrasp_ocid_vlg_stage1 maplegrasp_ocid_vlg_stage2 maplegrasp_vcot_stage1 maplegrasp_vcot_stage2 maplegrasp_realvlg_stage1 maplegrasp_realvlg_stage2)
CONFIGS=(config/experiments/smoke/graspmamba_ocid_vlg.yaml config/experiments/smoke/graspmamba_vcot.yaml config/experiments/smoke/graspmamba_realvlg.yaml config/experiments/smoke/maplegrasp_ocid_vlg_stage1.yaml config/experiments/smoke/maplegrasp_ocid_vlg_stage2.yaml config/experiments/smoke/maplegrasp_vcot_stage1.yaml config/experiments/smoke/maplegrasp_vcot_stage2.yaml config/experiments/smoke/maplegrasp_realvlg_stage1.yaml config/experiments/smoke/maplegrasp_realvlg_stage2.yaml)
STAGE1_REFS=("" "" "" "" maplegrasp_ocid_vlg_stage1 "" maplegrasp_vcot_stage1 "" maplegrasp_realvlg_stage1)

wait_for_all_npus() {
  while true; do
    if pgrep -af '[t]orchrun' >/dev/null || pgrep -af '[t]rain.py' >/dev/null; then
      sleep "${POLL_SECONDS}"
      continue
    fi
    local free_npus
    free_npus="$(npu-smi info 2>/dev/null | grep -c 'No running processes found in NPU' || true)"
    [[ "${free_npus}" -eq 8 ]] && return 0
    sleep "${POLL_SECONDS}"
  done
}

verify_checkpoint() {
  "${PYTHON}" -c 'import sys, torch; p=sys.argv[1]; c=torch.load(p, map_location="cpu"); assert isinstance(c, dict) and "state_dict" in c; print("CHECKPOINT_OK", p, "epoch", c.get("epoch"))' "$1"
}

run_stage() {
  local index="$1"
  local name="${NAMES[${index}]}"
  local config="${CONFIGS[${index}]}"
  local stage1_ref="${STAGE1_REFS[${index}]}"
  local exp_name="smoke_${RUN_ID}_${name}"
  local exp_dir="exp/smoke/${exp_name}"
  local launcher_log="${exp_dir}/launcher.smoke.log"
  local -a options=(TRAIN.exp_name "${exp_name}")

  wait_for_all_npus
  [[ -f "${config}" ]] || { echo "MISSING_CONFIG name=${name} config=${config}"; return 20; }
  if [[ -n "${stage1_ref}" ]]; then
    local stage1_checkpoint="exp/smoke/smoke_${RUN_ID}_${stage1_ref}/last.pth"
    [[ -f "${stage1_checkpoint}" ]] || { echo "MISSING_STAGE1_CHECKPOINT name=${name} checkpoint=${stage1_checkpoint}"; return 21; }
    options+=(TRAIN.weight "${stage1_checkpoint}")
  fi

  mkdir -p "${exp_dir}"
  echo "STAGE_START name=${name} time=$(date --iso-8601=seconds) config=${config} exp=${exp_name}"
  torchrun_guard_start "${launcher_log}" "${TORCHRUN}" --standalone --nproc_per_node=8 train.py --config "${config}" --opts "${options[@]}"
  local rc=0
  torchrun_guard_wait || rc=$?
  echo "STAGE_END name=${name} time=$(date --iso-8601=seconds) rc=${rc} log=${launcher_log}"
  [[ "${rc}" -eq 0 ]] || return "${rc}"

  grep -q 'Training time' "${launcher_log}" || { echo "MISSING_TRAINING_TIME name=${name}"; return 22; }
  grep -Eq 'Evaluation: Epoch=|RealVLG .*n=' "${launcher_log}" || { echo "MISSING_VALIDATION name=${name}"; return 23; }
  if grep -Eqi 'out of memory|(^|[^[:alpha:]])nan([^[:alpha:]]|$)|hccl.*(error|fail)|traceback|state_dict.*(error|mismatch)' "${launcher_log}"; then
    echo "HARD_ERROR_PATTERN name=${name}"
    return 24
  fi
  if [[ -n "${stage1_ref}" ]]; then
    grep -q 'Loaded initial model weight' "${launcher_log}" || { echo "MISSING_STAGE1_LOAD_MARKER name=${name}"; return 25; }
  fi
  verify_checkpoint "${exp_dir}/last.pth"
}

echo "QUEUE_START run_id=${RUN_ID} jobs=${#NAMES[@]} time=$(date --iso-8601=seconds)"
failures=()
for index in "${!NAMES[@]}"; do
  if run_stage "${index}"; then
    echo "STAGE_VERIFIED name=${NAMES[${index}]}"
  else
    rc=$?
    failures+=("${NAMES[${index}]}:${rc}")
    echo "STAGE_FAILED_CONTINUING name=${NAMES[${index}]} rc=${rc}"
  fi
done
if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "SMOKE_COMPLETE_WITH_FAILURES run_id=${RUN_ID} failures=${failures[*]} time=$(date --iso-8601=seconds)"
  exit 1
fi
echo "ALL_SMOKE_COMPLETE run_id=${RUN_ID} time=$(date --iso-8601=seconds)"
