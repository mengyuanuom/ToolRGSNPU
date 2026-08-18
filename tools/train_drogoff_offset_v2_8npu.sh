#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

dataset="${1:-}"
if [[ -z "${dataset}" ]]; then
  printf 'Usage: %s {grasp_tools|ocid_vlg|realvlg|vcot} [train.py arguments]\n' "$0" >&2
  exit 2
fi
shift

case "${dataset}" in
  grasp_tools|grasp-tools)
    config_path="config/grasp_tools/drogoff_offset_v2.yaml"
    ;;
  ocid_vlg|ocid-vlg)
    config_path="config/ocid_vlg/drogoff_offset_v2.yaml"
    ;;
  realvlg|graspnet_vlg|graspnet-vlg)
    config_path="config/realvlg/drogoff_offset_v2.yaml"
    ;;
  vcot|grasp-anything)
    config_path="config/vcot/drogoff_offset_v2.yaml"
    ;;
  *)
    printf 'Unknown dataset: %s\n' "${dataset}" >&2
    printf 'Choose grasp_tools, ocid_vlg, realvlg, or vcot.\n' >&2
    exit 2
    ;;
esac

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1

exec torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29620}" \
  train.py --config "${config_path}" "$@"
