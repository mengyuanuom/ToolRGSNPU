# GraspMamba on Ascend NPU

ToolRGSNPU keeps the official MambaVision model structure and pretrained
parameter names, but replaces its CUDA-oriented execution path at runtime.
The adapter does not substitute a CNN backbone under the GraspMamba name.

## What is replaced

MambaVision 1.2.0 imports `mamba_ssm.ops.selective_scan_interface`, whose
published dependency builds `selective_scan_cuda`. The NPU adapter provides:

- a pure-PyTorch selective scan using an associative parallel prefix scan;
- an optional sequential reference scan for diagnosis;
- activation checkpointing around the scan during training;
- depthwise Conv1d expressed as NPU-supported depthwise Conv2d; and
- the upstream attention module's non-fused matmul/softmax path.

These replacements add no learnable parameters. Official MambaVision-T
checkpoints therefore retain the same state-dict contract.

## Installation

Install CANN and a mutually matching `torch`/`torch_npu`/`torchvision` set
first. The script deliberately uses `--no-deps` so pip cannot replace that
Ascend stack. Then run:

```bash
bash tools/install_graspmamba_npu.sh
python tools/download_pretrained.py clip-rn50 mambavision-t
python tools/check_graspmamba_env.py
```

The checker runs a selective-scan forward/backward gradient test, verifies that
the runtime patch leaves every state-dict key unchanged, and then executes one
complete MambaVision-T forward pass on NPU 0.

The installation script installs `mambavision==1.2.0` with `--no-deps` so pip
does not attempt to compile CUDA `mamba-ssm`. ToolRGSNPU supplies the narrow
operator import surface used by MambaVision itself.

## Training

Eight-NPU OCID-VLG:

```bash
torchrun --nproc_per_node=8 train.py \
  --config config/ocid_vlg/graspmamba.yaml
```

Eight-NPU Grasp-Tools:

```bash
torchrun --nproc_per_node=8 train.py \
  --config config/grasp_tools/graspmamba.yaml
```

Two-NPU VCoT/Grasp-Anything:

```bash
torchrun --nproc_per_node=2 train.py \
  --config config/vcot/graspmamba.yaml
```

All checked-in profiles use:

```yaml
TRAIN:
  mamba_npu_fallback: True
  mamba_npu_scan_backend: parallel
  mamba_npu_scan_checkpoint: True
  amp: False
  sync_bn: False
```

Use `mamba_npu_scan_backend: sequential` only to diagnose numerical or operator
issues. It uses much less temporary memory but launches one recurrence step per
token and is substantially slower. If the parallel path runs out of memory,
first reduce the global training and validation batch sizes; keep them divisible
by the number of NPU processes.

## Validation boundary

The portable scan follows the selective state-space recurrence and supports
autograd, but it is not NVIDIA's fused CUDA kernel and its floating-point
reduction order differs. Before a long run, the environment checker must pass
on the exact CANN/torch_npu stack. Compare one fixed validation batch against a
GPU run when strict cross-device parity is required.

MambaVision source is distributed under NVIDIA's non-commercial source license,
and its pretrained weights use CC-BY-NC-SA-4.0. Review those licenses before
redistribution or commercial use.
