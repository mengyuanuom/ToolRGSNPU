#!/usr/bin/env bash
set -Eeo pipefail

PROJECT=/data1/ma00959358/pangu/ToolRGSNPU
TORCHRUN=/root/miniconda3/envs/pangu_mmy/bin/torchrun
PYTHON=/root/miniconda3/envs/pangu_mmy/bin/python
ASCEND_ENV=/data1/wangxuefei/Ascend/ascend-toolkit/set_env.sh
RUN_ID="${RUN_ID:-m4_grasp_aware_$(date +%Y%m%d_%H%M%S)}"
POLL_SECONDS="${POLL_SECONDS:-10}"
START_INDEX="${START_INDEX:-0}"
VAL_GLOBAL_BATCH="${VAL_GLOBAL_BATCH:-8}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
cd "${PROJECT}"
source "${ASCEND_ENV}"
set -u
source "${PROJECT}/tools/lib/torchrun_process_guard.sh"
torchrun_guard_install

NAMES=(ocid_vlg vcot realvlg grasp_tools_v3)
CONFIGS=(
  config/experiments/grasp_alignment_v1/ocid_vlg.yaml
  config/experiments/grasp_alignment_v1/vcot.yaml
  config/experiments/grasp_alignment_v1/realvlg.yaml
  config/experiments/grasp_alignment_v1/grasp_tools_v3.yaml
)
OUTPUT_ROOTS=(exp/ocid_vlg exp/vcot exp/realvlg exp/grasp_tools)
BATCH_CANDIDATES=(256 128)

OOM_PATTERN='out of memory|OutOfMemoryError|ACL_ERROR_RT_MEMORY_ALLOCATION|memory allocation[^[:cntrl:]]*(fail|error)'
HARD_ERROR_PATTERN='(^|[^[:alpha:]])nan([^[:alpha:]]|$)|(^|[^[:alpha:]])inf(inity)?([^[:alpha:]]|$)|hccl[^[:cntrl:]]*(error|fail)|traceback|state_dict[^[:cntrl:]]*(error|mismatch)'

[[ "${START_INDEX}" =~ ^[0-9]+$ && "${START_INDEX}" -lt "${#NAMES[@]}" ]] || {
  echo "INVALID_START_INDEX value=${START_INDEX} jobs=${#NAMES[@]}"
  exit 30
}
[[ "${VAL_GLOBAL_BATCH}" =~ ^[0-9]+$ && "$((VAL_GLOBAL_BATCH % 8))" -eq 0 ]] || {
  echo "INVALID_VALIDATION_BATCH value=${VAL_GLOBAL_BATCH}"
  exit 31
}

project_processes() {
  pgrep -af '/data1/ma00959358/pangu/ToolRGSNPU/.*(torchrun|train.py)|torchrun.*ToolRGSNPU|train.py.*grasp_alignment_v1' || true
}

all_npus_are_free() {
  local npu_info
  npu_info="$(npu-smi info 2>/dev/null)" || return 1
  ! grep -Eq '^\|[[:space:]]*[0-7][[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]*[0-9]+[[:space:]]+\|' <<<"${npu_info}"
}

wait_for_all_npus() {
  while true; do
    if [[ -z "$(project_processes)" ]] && all_npus_are_free; then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

verify_v3_dataset() {
  "${PYTHON}" -c 'import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = {"train": 12000, "val": 1000, "test": 2000}
meta_path = root / "metadata.json"
assert meta_path.is_file(), f"missing {meta_path}"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
for split, count in expected.items():
    split_dir = root / split
    images = list(split_dir.glob("*.jpg"))
    annotations = list(split_dir.glob("*.json"))
    index_path = split_dir / "index.jsonl"
    assert len(images) == count, (split, "images", len(images), count)
    assert len(annotations) == count, (split, "annotations", len(annotations), count)
    assert index_path.is_file() and index_path.stat().st_size > 0, index_path
    assert int(meta["stats"]["scenes"][split]) == count, (split, meta["stats"]["scenes"])
print("V3_DATASET_OK", root, expected)' \
    "${PROJECT}/datasets/grasp-tools/aug_graspall_v3_15k"
}

verify_checkpoint() {
  "${PYTHON}" -c 'import sys, torch
p = sys.argv[1]
c = torch.load(p, map_location="cpu", weights_only=False)
required = {"state_dict", "optimizer", "scheduler"}
assert isinstance(c, dict), type(c)
assert required.issubset(c), (p, sorted(required - set(c)))
print("CHECKPOINT_OK", p, "epoch", c.get("epoch"))' "$1"
}

verify_launch_topology() {
  local deadline=$((SECONDS + 120))
  local rank_count=0
  while (( SECONDS < deadline )); do
    kill -0 "${TORCHRUN_GUARD_ACTIVE_PID}" 2>/dev/null || break
    rank_count="$(pgrep -P "${TORCHRUN_GUARD_ACTIVE_PID}" -f 'train.py' 2>/dev/null | wc -l)"
    if [[ "${rank_count}" -eq 8 ]]; then
      echo "TOPOLOGY_OK torchrun_pid=${TORCHRUN_GUARD_ACTIVE_PID} direct_ranks=8"
      return 0
    fi
    sleep 2
  done
  echo "TOPOLOGY_INVALID torchrun_pid=${TORCHRUN_GUARD_ACTIVE_PID} direct_ranks=${rank_count}"
  return 40
}

verify_success_gate() {
  local name="$1" launcher_log="$2" checkpoint="$3" batch="$4"
  grep -q 'Training time' "${launcher_log}" || { echo "MISSING_TRAINING_TIME name=${name}"; return 22; }
  grep -Eq 'Evaluation: Epoch=|RealVLG .*n=' "${launcher_log}" || { echo "MISSING_VALIDATION name=${name}"; return 23; }
  if grep -Eqi "${OOM_PATTERN}|${HARD_ERROR_PATTERN}" "${launcher_log}"; then
    echo "HARD_ERROR_PATTERN name=${name} batch=${batch}"
    return 24
  fi
  grep -q "Batch sizes: train global=${batch} per-process=$((batch / 8)); validation global=${VAL_GLOBAL_BATCH} per-process=$((VAL_GLOBAL_BATCH / 8)); world_size=8" "${launcher_log}" || {
    echo "BATCH_MARKER_MISMATCH name=${name} batch=${batch}"
    return 25
  }
  verify_checkpoint "${checkpoint}"
}

run_attempt() {
  local index="$1" batch="$2"
  local name="${NAMES[${index}]}" config="${CONFIGS[${index}]}" output_root="${OUTPUT_ROOTS[${index}]}"
  local exp_name="formal_${RUN_ID}_m4_${name}_b${batch}"
  local exp_dir="${output_root}/${exp_name}" launcher_log="${exp_dir}/launcher.formal.log"
  local -a options=(TRAIN.exp_name "${exp_name}" TRAIN.batch_size "${batch}" TRAIN.batch_size_val "${VAL_GLOBAL_BATCH}" TRAIN.gradient_accumulation_steps 1)

  wait_for_all_npus
  [[ -f "${config}" ]] || { echo "MISSING_CONFIG name=${name} config=${config}"; return 20; }
  [[ ! -e "${exp_dir}" ]] || { echo "EXPERIMENT_EXISTS name=${name} exp_dir=${exp_dir}"; return 26; }
  mkdir -p "${exp_dir}"

  echo "ATTEMPT_START name=${name} time=$(date --iso-8601=seconds) config=${config} exp=${exp_name} global_batch=${batch} per_rank=$((batch / 8)) val_global=${VAL_GLOBAL_BATCH} old_checkpoint=none"
  torchrun_guard_start "${launcher_log}" "${TORCHRUN}" --standalone --nproc_per_node=8 train.py --config "${config}" --opts "${options[@]}"
  if ! verify_launch_topology; then
    torchrun_guard_stop
    return 40
  fi

  local rc=0
  torchrun_guard_wait || rc=$?
  echo "ATTEMPT_END name=${name} time=$(date --iso-8601=seconds) rc=${rc} batch=${batch} log=${launcher_log}"
  if grep -Eqi "${OOM_PATTERN}" "${launcher_log}"; then
    echo "ATTEMPT_OOM name=${name} batch=${batch}"
    return 42
  fi
  [[ "${rc}" -eq 0 ]] || return "${rc}"
  verify_success_gate "${name}" "${launcher_log}" "${exp_dir}/last.pth" "${batch}"
}

verify_v3_dataset
[[ -z "$(project_processes)" ]] || { echo "PROJECT_ALREADY_RUNNING"; project_processes; exit 32; }

echo "QUEUE_START run_id=${RUN_ID} jobs=${#NAMES[@]} start_index=${START_INDEX} batch_policy=256_then_128_on_oom val_global=${VAL_GLOBAL_BATCH} time=$(date --iso-8601=seconds)"
for ((index=START_INDEX; index<${#NAMES[@]}; index++)); do
  name="${NAMES[${index}]}"
  completed=0
  for batch in "${BATCH_CANDIDATES[@]}"; do
    if run_attempt "${index}" "${batch}"; then
      echo "STAGE_VERIFIED name=${name} batch=${batch}"
      completed=1
      break
    fi
    rc=$?
    if [[ "${rc}" -eq 42 && "${batch}" -eq 256 ]]; then
      echo "OOM_FALLBACK name=${name} from_global=256 to_global=128 time=$(date --iso-8601=seconds)"
      wait_for_all_npus
      continue
    fi
    echo "STAGE_FAILED_STOPPING name=${name} batch=${batch} rc=${rc} time=$(date --iso-8601=seconds)"
    exit "${rc}"
  done
  [[ "${completed}" -eq 1 ]] || { echo "STAGE_EXHAUSTED_BATCHES name=${name}"; exit 43; }
done
echo "ALL_M4_GRASP_ALIGNMENT_COMPLETE run_id=${RUN_ID} time=$(date --iso-8601=seconds)"
