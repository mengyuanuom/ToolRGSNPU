# Running ToolRGSNPU on Ascend

ToolRGSNPU uses explicit `torch_npu` APIs. It does not use
`torch_npu.contrib.transfer_to_npu`, so CUDA calls are not silently monkey
patched and device errors retain useful stack traces.

## 1. Install the Ascend stack

Install the CANN toolkit/runtime and source its environment script before
starting Python. Install a PyTorch and `torch_npu` pair listed as compatible
with that exact CANN release. The versions are intentionally not pinned in this
repository because a wheel for one CANN release is not interchangeable with
another.

Use the official Ascend adapter release and installation table:
<https://gitee.com/ascend/pytorch>.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -r requirement-npu.txt
python tools/check_npu_env.py
```

The check must report an NPU device, HCCL availability, and a successful NPU
matrix multiplication before a training run is attempted.

Training, evaluation, and deployment run an artifact preflight before model
construction. If a configured weight is missing, the log reports its config
key, original value, resolved absolute path, and current working directory.

## 2. Put datasets and pretrained weights in place

Datasets, CLIP, DINOv2, MambaVision, and trained checkpoints are not committed
to Git. Set the paths in the selected experiment YAML. Validate them without
allocating the full model:

```bash
python tools/check_npu_env.py --config config/vcot/drogoff.yaml
```

Build the model and run a one-sample forward pass:

```bash
python tools/check_npu_env.py --config config/vcot/drogoff.yaml --forward
```

## 3. Train

Single NPU:

```bash
python train.py --config config/vcot/drogoff.yaml
```

Two NPUs on one machine:

```bash
torchrun --nproc_per_node=2 train.py --config config/vcot/drogoff.yaml
```

`LOCAL_RANK` selects the local NPU and the process group always uses HCCL.
Batch sizes in YAML are per NPU. Start conservatively and increase them after
the model smoke test succeeds.

Use standard Adam by default. To test the Ascend fused optimizer, set:

```yaml
TRAIN:
  optimizer: npu_fused_adam
```

The installed `torch_npu` build must expose `torch_npu.optim.NpuFusedAdam`.

## 4. Compatibility boundary

The ToolRGS engine, datasets, dense losses, validation, offset refinement,
CLIP backbones, DINOv2 fallback attention, CROG, CROG-OFF, DROG, DROG-OFF,
GGCNN-CLIP, GR-ConvNet-CLIP, and LGD use portable PyTorch operators and are
wired to NPU/HCCL.

GraspMamba is retained in the registry and configs, but its upstream
MambaVision package depends on the CUDA `selective_scan_cuda` extension. It is
therefore experimental on Ascend and requires an Ascend-native selective-scan
implementation from that dependency. ToolRGSNPU does not substitute a
different convolutional network under the GraspMamba name because that would
invalidate comparisons.

The optional MMDetection GUI tab additionally requires an NPU-compatible
MMCV/MMDetection installation. Whisper remains on CPU by default. Camera and
robot I/O are CPU operations.

## 5. Common diagnostics

- `torch_npu` import errors mentioning `libascendcl.so` or `libhccl.so`: source
  the CANN environment and confirm toolkit/runtime packages are installed.
- HCCL initialization errors: verify visible devices, rank variables, and that
  every process uses a distinct `LOCAL_RANK`.
- Unsupported operator errors: rerun with one NPU and AMP disabled
  (`TRAIN.amp False`) to identify the exact operator and dtype.
- Out of memory: reduce both `TRAIN.batch_size` and `TRAIN.batch_size_val`.
