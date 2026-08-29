#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -c 'import torch, torch_npu, torchvision' >/dev/null 2>&1 || {
  echo "[GraspMamba] install matching torch, torch_npu, and torchvision wheels first." >&2
  exit 2
}

python -m pip install --no-deps -r requirement-mamba.txt
# Do not let MambaVision pull mamba-ssm: its published wheel/source builds a
# CUDA selective_scan extension. ToolRGSNPU supplies the compatible operator
# import surface and NPU forward path at runtime.
python -m pip install --no-deps mambavision==1.2.0

python -c 'import einops, mambavision, safetensors, timm' >/dev/null

echo "[GraspMamba] dependencies installed. Run the NPU smoke check:"
echo "python tools/check_graspmamba_env.py"
