# RealVLG-R1 compatibility guide

This adapter targets the `GraspNet_VLG` portion of RealVLG-11B because the
public RealVLG-R1 evaluator currently exposes its executable
seen/similar/novel protocol for that subset.

Official resources:

- Paper: <https://arxiv.org/abs/2603.14880>
- Evaluation code: <https://github.com/lif314/RealVLG-R1>
- Dataset: <https://modelscope.cn/datasets/cslinfeili/RealVLG-11B>

## Data

Download `GraspNet_VLG.zip` from the official RealVLG-11B dataset page. The
published LFS manifest declares a size of 109,726,723,207 bytes. Extract it so
that the configured root directly contains `metadata/`:

```text
/path/to/GraspNet_VLG/
├── metadata/
│   └── kinect/
│       ├── scene_0000/
│       │   ├── 0000.json
│       │   └── ...
│       └── scene_0189/
├── <RGB image paths referenced by metadata>
└── <mask paths referenced by metadata>
```

Each JSON object must provide `image_path`, `mask_path`, `description`,
`bbox`, and one or more 8-coordinate `grasps`. Paths are resolved relative to
the configured dataset root.

The default profile is:

```yaml
DATA:
  dataset: realvlg
  root_path: ./datasets/GraspNet_VLG
  with_depth: false
  train_split: train
  val_split: seen
```

RealVLG's public metadata contract does not provide a matching `depth_path`;
the adapter therefore rejects `with_depth: true`.

## Official splits

The implementation follows the ranges in the public
`evaluation/dataset.py`, rather than its stale docstring:

| Split | Scenes | Frames evaluated | Samples |
| --- | --- | --- | --- |
| `seen` | 0100–0129 | `0000.json` only | every object with nonempty grasps |
| `similar` | 0130–0159 | `0000.json` only | every object with nonempty grasps |
| `novel` | 0160–0189 | `0000.json` only | every object with nonempty grasps |

Training scans scenes 0000–0099. The paper's benchmark experiments use 10% of
the training data and 10 epochs, which are the defaults in
`config/realvlg/drogoff.yaml`.

The public repository refers to an author-local
`train_gs_kn_10p.parquet`, but does not publish the selected row identifiers.
The adapter therefore selects exactly 10% of sample keys by a stable SHA-256
ranking for a deterministic substitute. It must not be described as the same 10% rows as
the paper. If the official identifiers become available, place one key per
line in a manifest:

```text
scene_0000/0000.json#3
scene_0000/0001.json#8
```

Then set `DATA.dataset_args.train_manifest` to that file. A manifest overrides
the deterministic fraction selector.

## Image and annotation preprocessing

The following behavior matches the published data/evaluation contract:

- RGB is decoded without random crop, flip, rotation, or color augmentation.
- Ground-truth masks are decoded in grayscale and binarized with `value > 128`.
- Referring descriptions are used verbatim and tokenized with CLIP's full
  77-token context.
- All grasp corner coordinates are retained in the untouched source-image
  coordinate system for evaluation.
- The four grasp corners are transformed before dense quality, doubled-angle
  sine/cosine, width, and DROG-OFF offset supervision is generated.
- Predictions are mapped back through the inverse affine transform before any
  benchmark metric is computed.

There is one unavoidable model-specific difference. The official Qwen2.5-VL
pipeline preserves aspect ratio and constrains image area to
262,144–4,194,304 pixels. ToolRGS's convolutional decoder requires one fixed
spatial size and CLIP normalization, so the adapter uses a deterministic
aspect-preserving letterbox to 448×448. It never stretches or crops the image,
and evaluation happens after exact inverse mapping to the original resolution.
This is the closest geometry-preserving ToolRGS preprocessing, but it is not
the Qwen processor itself.

The official evaluator also wraps the description in a Qwen reasoning/output
format prompt. ToolRGS has no autoregressive answer parser, so its CLIP text
encoder receives the underlying RealVLG description directly.

## What is and is not identical to the paper

| Component | Compatibility |
| --- | --- |
| Public JSON annotations and mask threshold | Exact |
| Executable scene ranges and `0000.json` test sampling | Exact |
| Original-coordinate evaluation and all metric formulas | Exact |
| Fixed 40-pixel predicted gripper depth and strict grasp thresholds | Exact |
| The paper's exact 10% training rows | Not publicly recoverable; deterministic exact-size substitute or manifest override |
| Qwen adaptive-resolution processor | Replaced by geometry-preserving 448×448 ToolRGS letterbox |
| Qwen reasoning prompt and GRPO/GSPO optimization | Replaced by ToolRGS CLIP text input and supervised multitask losses |
| Qwen box + frozen SAM2 segmentation path | Replaced by ToolRGS's direct mask prediction |

The resulting experiment is therefore a fair ToolRGS evaluation on the public
RealVLG protocol, not a reproduction of the RealVLG-R1 model architecture.

## Official evaluation metrics

Segmentation uses the public implementations:

- `mean_gIoU` and `mean_cIoU`: computed from the tight box of ToolRGS's
  predicted mask and the annotated RealVLG box.
- `F_beta`: the public code computes the ordinary F1 score.
- `S_alpha`: `alpha=0.5`, combining foreground/background agreement and four
  fixed image quadrants.
- `E_measure`: mean enhanced-alignment score.
- `Segmentation_Validity_Rate`: finite, nonempty mask predictions divided by
  all evaluated samples; an empty mask cannot produce the required box.

For grasping, exactly one prediction `(x, y, theta, width)` is evaluated:

1. Construct its rectangle using a fixed 40-pixel gripper depth.
2. Compare it with every ground-truth 8-point grasp using continuous polygon
   IoU.
3. Retain the ground truth with maximum IoU.
4. Report `mIoU` over valid predictions.
5. Mark it correct only when `best_IoU > 0.25` **and**
   `best_angle_difference < 30 degrees`.
6. Report `gAcc` and `Grasp_Validity_Rate`.

The official evaluator's unusual small-angle radians auto-detection is retained
verbatim for result compatibility.

ToolRGS directly predicts a mask, whereas the paper's Qwen baseline predicts a
box and uses frozen SAM2 to produce its mask. F/S/E remain the same dataset
metrics; direct-mask results should be reported as ToolRGS rather than as the
Qwen+SAM2 inference pipeline.

Each evaluation writes an aggregate JSON file under the experiment directory,
for example `realvlg_seen_metrics.json`.

## Commands

Download the required CLIP and DINOv2 weights once:

```bash
python tools/download_pretrained.py clip-vit-b16 dinov2-vitb14-reg4
```

Train DROG-OFF on eight NPUs:

```bash
torchrun --nproc_per_node=8 train.py \
  --config config/realvlg/drogoff.yaml --opts \
  DATA.root_path /path/to/GraspNet_VLG
```

To use the lightweight grasp-region Offset V2 head under the same official
data and evaluation protocol, replace the config with:

```bash
torchrun --nproc_per_node=8 train.py \
  --config config/realvlg/drogoff_offset_v2.yaml --opts \
  DATA.root_path /path/to/GraspNet_VLG
```

Evaluate one official split on one NPU:

```bash
python evaluate.py \
  --config config/realvlg/drogoff.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --npu 0 --opts \
  DATA.root_path /path/to/GraspNet_VLG \
  TEST.test_split seen
```

Evaluate `seen`, `similar`, and `novel` sequentially:

```bash
bash tools/evaluate_realvlg_splits.sh \
  config/realvlg/drogoff.yaml \
  /path/to/checkpoint.pth \
  /path/to/GraspNet_VLG
```
