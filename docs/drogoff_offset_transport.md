# DrogOff Offset-Transport

This experiment promotes the existing dense Offset V2 prediction from an
auxiliary output to a grasp-feature routing signal. It deliberately keeps the
historical DINOv2/CLIP adapters and DETRIS query decoder for a controlled first
test.

For every OS4 grasp-branch feature at pixel `x`, the module samples center
context at `x + max_displacement * offset(x)`. A quality-conditioned residual
injects this context into quality, sine, cosine, width, and optional short-side
features. The segmentation branch never enters the transport module.

Each grasp branch owns a `tanh` residual gate initialized to exactly zero.
Consequently, the first forward pass matches Offset V2 DrogOff, while gradient
descent can independently open the gates. The coarse quality confidence is
detached by default to avoid a self-reinforcing routing shortcut.

## OCID-VLG training

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --config config/experiments/ocid_vlg/drogoff_transport.yaml
```

To warm-start from a matching Offset V2 checkpoint, set `TRAIN.weight` in the
experiment file or override it on the command line. Do not use an Offset V1 or
LoRA checkpoint.

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --config config/experiments/ocid_vlg/drogoff_transport.yaml \
  --opts TRAIN.weight /path/to/offset_v2_checkpoint.pth
```

Resume only from this experiment's own `last.pth`, because resume loading is
strict and includes the transport parameters.

## Recommended first ablation

Compare the unchanged `config/ocid_vlg/drogoff_offset_v2.yaml` profile against
this profile under the same seed, batch, split, and evaluation protocol. Inspect
the learned `proj.transport.branch_gate` values; gates remaining near zero are
evidence that the corresponding grasp quantity did not benefit from transport.
