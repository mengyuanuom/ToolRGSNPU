#!/usr/bin/env bash
set -eo pipefail

PROJECT=/data1/ma00959358/pangu/ToolRGSNPU
TORCHRUN=/root/miniconda3/envs/pangu_mmy/bin/torchrun
PYTHON=/root/miniconda3/envs/pangu_mmy/bin/python
ASCEND_ENV=/data1/wangxuefei/Ascend/ascend-toolkit/set_env.sh

cd "$PROJECT"
# Ascend's setup script reads optional variables before assigning defaults.
# Load it before enabling nounset for the queue itself.
# shellcheck disable=SC1090
source "$ASCEND_ENV"
set -u

wait_for_all_npus() {
  while true; do
    if pgrep -af '[t]orchrun' >/dev/null; then
      sleep 60
      continue
    fi
    free_npus=$(npu-smi info 2>/dev/null | grep -c 'No running processes found in NPU' || true)
    if [[ "$free_npus" -eq 8 ]]; then
      return 0
    fi
    sleep 60
  done
}

next_log() {
  local exp_dir=$1
  local candidate="$exp_dir/launcher.train.log"
  if [[ -e "$candidate" ]]; then
    candidate="$exp_dir/launcher.train.$(date +%Y%m%d_%H%M%S).log"
  fi
  printf '%s\n' "$candidate"
}

verify_checkpoint() {
  local checkpoint=$1
  "$PYTHON" -c 'import sys, torch; p=sys.argv[1]; c=torch.load(p, map_location="cpu"); assert isinstance(c, dict); print("CHECKPOINT_OK", p, "epoch", c.get("epoch"))' "$checkpoint"
}

run_stage() {
  local name=$1
  local config=$2
  local exp_dir=$3

  wait_for_all_npus
  if [[ -e "$exp_dir/last.pth" ]]; then
    echo "REFUSE_EXISTING_CHECKPOINT $name $exp_dir/last.pth"
    return 20
  fi

  mkdir -p "$exp_dir"
  local launcher_log
  launcher_log=$(next_log "$exp_dir")
  echo "STAGE_START $name $(date --iso-8601=seconds) config=$config log=$launcher_log"

  set +e
  "$TORCHRUN" --standalone --nproc_per_node=8 train.py --config "$config" \
    >"$launcher_log" 2>&1
  local rc=$?
  set -e

  echo "STAGE_END $name $(date --iso-8601=seconds) rc=$rc log=$launcher_log"
  if [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi
  if ! grep -q 'Training time' "$launcher_log"; then
    echo "MISSING_TRAINING_TIME $name $launcher_log"
    return 21
  fi
  if grep -Eqi 'out of memory|(^|[^[:alpha:]])nan([^[:alpha:]]|$)|hccl.*(error|fail)|traceback|state_dict.*(error|mismatch)' "$launcher_log"; then
    echo "HARD_ERROR_PATTERN $name $launcher_log"
    return 22
  fi
  verify_checkpoint "$exp_dir/last.pth"
}

run_stage realvlg \
  config/experiments/realvlg/drogoff_transport_accum2.yaml \
  exp/realvlg/drogoff_transport_realvlg_full_e24_8npu_bs256

run_stage ocid_vlg \
  config/experiments/ocid_vlg/drogoff_transport.yaml \
  exp/ocid_vlg/drogoff_transport_ocid_vlg_8npu

run_stage vcot \
  config/experiments/vcot/drogoff_transport.yaml \
  exp/vcot/drogoff_transport_vcot_8npu_bs128_e36

echo "ALL_STAGES_COMPLETE $(date --iso-8601=seconds)"
