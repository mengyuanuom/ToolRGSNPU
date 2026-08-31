#!/usr/bin/env bash
set -Eeo pipefail

PROJECT=/data1/ma00959358/pangu/ToolRGSNPU
TORCHRUN=/root/miniconda3/envs/pangu_mmy/bin/torchrun
PYTHON=/root/miniconda3/envs/pangu_mmy/bin/python
ASCEND_ENV=/data1/wangxuefei/Ascend/ascend-toolkit/set_env.sh
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
POLL_SECONDS="${POLL_SECONDS:-10}"
START_INDEX="${START_INDEX:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
CONTINUATION_ID="${CONTINUATION_ID:-$(date +%Y%m%d_%H%M%S)}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
cd "${PROJECT}"
source "${ASCEND_ENV}"
set -u
source "${PROJECT}/tools/lib/torchrun_process_guard.sh"
torchrun_guard_install

NAMES=(graspmamba_ocid_vlg graspmamba_vcot graspmamba_realvlg maplegrasp_ocid_vlg_stage1 maplegrasp_ocid_vlg_stage2 maplegrasp_vcot_stage1 maplegrasp_vcot_stage2 maplegrasp_realvlg_stage1 maplegrasp_realvlg_stage2)
CONFIGS=(config/experiments/formal_six/graspmamba_ocid_vlg.yaml config/experiments/formal_six/graspmamba_vcot.yaml config/experiments/formal_six/graspmamba_realvlg.yaml config/experiments/formal_six/maplegrasp_ocid_vlg_stage1.yaml config/experiments/formal_six/maplegrasp_ocid_vlg_stage2.yaml config/experiments/formal_six/maplegrasp_vcot_stage1.yaml config/experiments/formal_six/maplegrasp_vcot_stage2.yaml config/experiments/formal_six/maplegrasp_realvlg_stage1.yaml config/experiments/formal_six/maplegrasp_realvlg_stage2.yaml)
OUTPUT_ROOTS=(exp/ocid_vlg exp/vcot exp/realvlg exp/ocid_vlg exp/ocid_vlg exp/vcot exp/vcot exp/realvlg exp/realvlg)
STAGE1_REFS=("" "" "" "" maplegrasp_ocid_vlg_stage1 "" maplegrasp_vcot_stage1 "" maplegrasp_realvlg_stage1)

[[ "${START_INDEX}" =~ ^[0-9]+$ ]] || {
  echo "INVALID_START_INDEX value=${START_INDEX}"
  exit 30
}
[[ "${START_INDEX}" -lt "${#NAMES[@]}" ]] || {
  echo "START_INDEX_OUT_OF_RANGE value=${START_INDEX} jobs=${#NAMES[@]}"
  exit 31
}

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
  "${PYTHON}" -c 'import sys, torch; p=sys.argv[1]; c=torch.load(p, map_location="cpu", weights_only=False); assert isinstance(c, dict) and "state_dict" in c; print("CHECKPOINT_OK", p, "epoch", c.get("epoch"))' "$1"
}

run_stage() {
  local index="$1"
  local name="${NAMES[${index}]}"
  local config="${CONFIGS[${index}]}"
  local output_root="${OUTPUT_ROOTS[${index}]}"
  local stage1_ref="${STAGE1_REFS[${index}]}"
  local exp_name="formal_${RUN_ID}_${name}"
  local exp_dir="${output_root}/${exp_name}"
  local launcher_log="${exp_dir}/launcher.formal.log"
  local -a options=(TRAIN.exp_name "${exp_name}")
  local is_resume=0

  wait_for_all_npus
  [[ -f "${config}" ]] || { echo "MISSING_CONFIG name=${name} config=${config}"; return 20; }
  if [[ "${index}" -eq "${START_INDEX}" && -n "${RESUME_CHECKPOINT}" ]]; then
    local expected_resume="${exp_dir}/last.pth"
    [[ "${RESUME_CHECKPOINT}" == "${expected_resume}" ]] || {
      echo "REFUSE_FOREIGN_RESUME name=${name} expected=${expected_resume} actual=${RESUME_CHECKPOINT}"
      return 27
    }
    [[ -f "${RESUME_CHECKPOINT}" ]] || {
      echo "MISSING_RESUME_CHECKPOINT name=${name} checkpoint=${RESUME_CHECKPOINT}"
      return 28
    }
    verify_checkpoint "${RESUME_CHECKPOINT}" || return 29
    launcher_log="${exp_dir}/launcher.resume.${CONTINUATION_ID}.log"
    options+=(TRAIN.resume "${RESUME_CHECKPOINT}")
    is_resume=1
  else
    [[ ! -e "${exp_dir}" ]] || { echo "EXPERIMENT_EXISTS name=${name} exp_dir=${exp_dir}"; return 26; }
  fi
  if [[ -n "${stage1_ref}" && "${is_resume}" -eq 0 ]]; then
    local stage1_checkpoint="${output_root}/formal_${RUN_ID}_${stage1_ref}/last.pth"
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
  if [[ -n "${stage1_ref}" && "${is_resume}" -eq 0 ]]; then
    grep -q 'Loaded initial model weight' "${launcher_log}" || { echo "MISSING_STAGE1_LOAD_MARKER name=${name}"; return 25; }
  fi
  if [[ "${is_resume}" -eq 1 ]]; then
    grep -q 'Resumed experiment from epoch' "${launcher_log}" || { echo "MISSING_RESUME_MARKER name=${name}"; return 32; }
  fi
  verify_checkpoint "${exp_dir}/last.pth"
}

echo "QUEUE_START run_id=${RUN_ID} jobs=${#NAMES[@]} start_index=${START_INDEX} resume=${RESUME_CHECKPOINT:-none} continuation_id=${CONTINUATION_ID} time=$(date --iso-8601=seconds)"
failures=()
for ((index=START_INDEX; index<${#NAMES[@]}; index++)); do
  if run_stage "${index}"; then
    echo "STAGE_VERIFIED name=${NAMES[${index}]}"
  else
    rc=$?
    failures+=("${NAMES[${index}]}:${rc}")
    echo "STAGE_FAILED_CONTINUING name=${NAMES[${index}]} rc=${rc}"
  fi
done
if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "FORMAL_COMPLETE_WITH_FAILURES run_id=${RUN_ID} failures=${failures[*]} time=$(date --iso-8601=seconds)"
  exit 1
fi
echo "ALL_FORMAL_COMPLETE run_id=${RUN_ID} time=$(date --iso-8601=seconds)"
