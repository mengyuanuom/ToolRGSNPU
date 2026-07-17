# ToolRGSNPU

Tool-oriented Referring Grasp Synthesis with a single configuration-driven
codebase for CROG, CROG-OFF, DROG, DROG-OFF, ETRG-A, MapleGrasp, GraspMamba,
LGD, GGCNN-CLIP, GR-ConvNet-CLIP, and DETRIS backbones. Grasp-Tools, VCoT/Grasp-Anything,
and OCID-VLG data use the same model-facing batch contract. This repository is
the Ascend NPU port: training and inference use `torch_npu`, AMP uses NPU AMP,
and distributed jobs use HCCL.

Start with the [Ascend installation and smoke-test guide](docs/ascend_npu.md).
The original CUDA project remains in `mengyuanuom/ToolRGS`.
This port was branched from ToolRGS commit `59fc3cc`.
The ETRG-A RGB-D integration is synchronized from ToolRGS commit `0c53ea0`.
The model/config matrix is synchronized through ToolRGS commit `a12f75e`.

## Design

All architectures live below `model/` and are selected by `MODEL.architecture`
in YAML. Datasets are selected by `DATA.dataset`; the registered adapters feed
one model-facing batch contract and one training/evaluation engine.

```text
ToolRGSNPU/
├── model/
│   ├── crog.py
│   ├── crogoff.py
│   ├── drog.py
│   ├── drogoff.py
│   ├── maplegrasp.py
│   ├── ggcnnclip.py
│   ├── grconvnetclip.py
│   ├── graspmamba.py
│   ├── lgd.py
│   ├── segmenter.py
│   └── dinov2/
├── config/grasp_tools/
├── engine/engine.py
├── utils/dataset.py
├── train.py
└── evaluate.py
```

`DROGOFF` combines DROG's DINOv2 + CLIP-adapter fusion with a two-channel
normalized center-offset head. Offset supervision is generated from transformed
Grasp-Tools rectangle centers and weighted by a Gaussian `off_w` map.

## Configuration

Choose a model entirely from the experiment config:

```yaml
MODEL:
  architecture: drogoff
```

New experiments use composable MMEngine-style `_base_` configs and the
registered NPU Runner. Model, dataset, schedule, and runtime settings are
independent files under `configs/_base_/`:

```yaml
_base_:
  - ../_base_/datasets/ocid_vlg.yaml
  - ../_base_/models/etrg_r50.yaml
  - ../_base_/schedules/etrg_40e.yaml
  - ../_base_/runtime/ascend.yaml

TRAIN:
  exp_name: etrg_r50_ocid_vlg
  output_folder: exp/ocid_vlg
```

Preferred training entrypoint:

```bash
python tools/train.py --config configs/etrg/etrg_r50_ocid_vlg.yaml
```

`NPUGraspRunner`, `NPUAmpOptimWrapper`, registered schedulers, and runner hooks
now own construction, AMP/backward, epoch scheduling, logging, and checkpoints.
The old `python train.py --config config/...` command remains compatible.

The nine RGB model families are available for every dataset. ETRG-A is added
for OCID-VLG because it requires real aligned depth:

| Dataset config directory | Models |
| --- | --- |
| `config/grasp_tools/` | `crog`, `crogoff`, `drog`, `drogoff`, `maplegrasp`, `ggcnnclip`, `grconvnetclip`, `graspmamba`, `lgd` |
| `config/vcot/` | `crog`, `crogoff`, `drog`, `drogoff`, `maplegrasp`, `ggcnnclip`, `grconvnetclip`, `graspmamba`, `lgd` |
| `config/ocid_vlg/` | `crog`, `crogoff`, `drog`, `drogoff`, `etrg`, `maplegrasp`, `ggcnnclip`, `grconvnetclip`, `graspmamba`, `lgd` |

For example, `config/vcot/drogoff.yaml` and
`config/ocid_vlg/lgd.yaml` are directly runnable after setting data and weight
paths. DETRIS remains a referring-segmentation baseline and does not implement
the shared grasp-map output/loss contract, so it is not included in this matrix.

Set `DATA.root_path`, `TRAIN.clip_pretrain`, and (for DROG variants)
`TRAIN.dino_pretrain` to local paths before training.

## Embedded Grasp-Tools v2 data and augmentation

The complete Grasp-Tools source set is included in this repository: 107
annotated RGB images with JSON masks/grasps and 38 background images live under
`assets/grasp_tools/`. Generate the multi-object, multi-query v2 dataset from a
fresh clone with:

```bash
python -u tools/dataset_converters/grasp_tools/augment.py \
  --image-ext jpg
```

The default output is `datasets/grasp-tools/aug_graspall_v2`. It contains
disjoint train/val/test background splits, balanced scale/angle sampling,
same-category and hard-negative scenes, and multiple referring expressions per
rendered image. Train the supplied NPU DROG-OFF experiment with:

```bash
python train.py --config config/grasp_tools/drogoff_v2.yaml
```

See `docs/grasp_tools_v2.md` for the smoke test, output schema, and full
generation options.

## Pretrained backbone weights

Official CLIP, DINOv2, MambaVision, and optional ETRG ResNet-18 download links
are collected in [docs/pretrained_weights.md](docs/pretrained_weights.md). The
repository also provides a dependency-free downloader. For DROG/DROG-OFF:

```bash
python tools/download_pretrained.py clip-vit-b16 dinov2-vitb14-reg4
```

For the RN50-based model families:

```bash
python tools/download_pretrained.py clip-rn50
```

These files initialize backbones. `TRAIN.weight` and `TRAIN.resume` refer to
task checkpoints produced by ToolRGS training and are documented separately in
the same guide.

## VCoT / Grasp-Anything data

The repository includes the small official seen/unseen test CSVs. The 27 MB
`train.csv` is intentionally excluded; download it directly from the official
VCoT-Grasp repository:

```bash
curl -L \
  https://raw.githubusercontent.com/zhanghr2001/VCoT-Grasp/main/split/vcot/train.csv \
  -o split/vcot/train.csv
```

The [official split directory](https://github.com/zhanghr2001/VCoT-Grasp/tree/main/split/vcot)
is the source of truth. Keep the large image, `.pt`, and `.npy` files outside
Git in this layout:

```text
/path/to/grasp-anything/
├── image/<scene_id>.jpg
├── positive_grasp/<grasp_id>.pt
└── mask/<grasp_id>.npy
```

Set the dataset in YAML:

```yaml
DATA:
  dataset: vcot
  root_path: /path/to/grasp-anything
  split_root: ./split/vcot
  train_split: train
  val_split: unseen       # or seen
  prompt_template: "Grasp the {object_name}"
```

The adapter reads `.pt` grasps as
`[score, x, y, width, height, theta_degrees]`, discards the score for geometry,
reorders the quadrilateral for ToolRGS's width/angle convention, and generates
grasp maps after letterboxing. Original-coordinate grasp
targets are retained for Jacquard evaluation. Files are loaded lazily per
sample; the dataset does not preload the full annotation corpus.

Inspect the same sample you verified previously:

```bash
python tools/inspect_vcot_sample.py \
  --dataset-root /mnt/ssd0/mengyuan/data/grasp-anything \
  --csv split/vcot/train.csv --row 2
```

All nine grasp-aware ToolRGS models can use VCoT without code changes. Use the
matching file under `config/vcot/`, for example:

```bash
python train.py --config config/vcot/drogoff.yaml --opts \
  DATA.root_path /mnt/ssd0/mengyuan/data/grasp-anything
```

### Initial VCoT profile for two Ascend NPUs

VCoT YAML batch sizes and worker counts are per distributed process (per NPU).
The inherited values are starting points, not published Ascend throughput or
memory benchmarks:

| Model | Input | Train batch/NPU | Global batch | Epochs | LR milestones |
| --- | ---: | ---: | ---: | ---: | --- |
| CROG / CROG-OFF | 416 | 8 | 16 | 70 | 55, 65 |
| MapleGrasp | 416 | 8 | 16 | 70 | 55, 65 |
| DROG / DROG-OFF | 448 | 8 | 16 | 65 | 35, 55 |
| GGCNN-CLIP | 416 | 32 | 64 | 50 | 35 |
| GRConvNet-CLIP | 416 | 32 | 64 | 80 | 70 |
| GraspMamba | 416 | 8 | 16 | 50 | 35, 45 |
| LGD | 224 | 16 | 32 | 100 | 70, 90 |

Each process uses eight training workers and four validation workers, producing
16 training workers across two NPUs. Start a two-NPU run with `torchrun`:

```bash
torchrun --nproc_per_node=2 train.py \
  --config config/vcot/graspmamba.yaml --opts \
  DATA.root_path /mnt/ssd0/mengyuan/data/grasp-anything
```

If a heavy model runs out of memory, reduce both per-NPU batches without
editing YAML, for example `TRAIN.batch_size 4 TRAIN.batch_size_val 4`.

## OCID-VLG data

OCID-VLG referring expressions are read directly from the downloaded dataset;
the large RGB, depth, and annotation files are not copied into this repository.
For every RGB-only ToolRGS model, download the compact
[OCID-VLG-S.zip release](https://github.com/mengyuanuom/ToolRGSNPU/releases/download/ocid-vlg-s-v1/OCID-VLG-S.zip)
(899.13 MiB, SHA256
`09f0e3f1c20c53de889f0ccd516f876f534199c3491d1e511726130471c44fd1`).
It keeps RGB, referring expressions, instance masks, and grasp annotations while
removing all depth and PCD files. ETRG-A users must instead download the full
`OCID-VLG.zip` from the [official repository](https://github.com/gtziafas/OCID-VLG)
or its [official Google Drive file](https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link).

For `/mnt/ssd0/mengyuan/ToolRGSNPU`, extract it to
`/mnt/ssd0/mengyuan/data/OCID-VLG`. Create a local `datasets -> ../data`
symlink once (it is ignored by Git), so the checked-in configs can keep
`DATA.root_path: ./datasets/OCID-VLG`:

```bash
mkdir -p /mnt/ssd0/mengyuan/data
cd /mnt/ssd0/mengyuan/data
curl -L \
  https://github.com/mengyuanuom/ToolRGSNPU/releases/download/ocid-vlg-s-v1/OCID-VLG-S.zip \
  -o OCID-VLG-S.zip
echo '09f0e3f1c20c53de889f0ccd516f876f534199c3491d1e511726130471c44fd1  OCID-VLG-S.zip' \
  | sha256sum -c -
unzip OCID-VLG-S.zip
cd /mnt/ssd0/mengyuan/ToolRGSNPU
ln -sfn ../data datasets
```

Use the extracted directory that directly contains `refer/` as the root.
The expected layout is:

```text
/path/to/OCID-VLG/
├── refer/multiple/
│   ├── train_expressions.json
│   ├── val_expressions.json
│   └── test_expressions.json
└── <sequence>/
    ├── rgb/<image_name>
    └── seg_mask_instances_combi/<image_name>
```

The full ETRG dataset additionally contains `<sequence>/depth/<image_name>`.

Use the supplied experiment or select the dataset from another model config:

```yaml
DATA:
  dataset: OCID-VLG
  root_path: /path/to/OCID-VLG
  version: multiple
  with_depth: false
  train_split: train
  val_split: val
```

Only ETRG-A sets `with_depth: true`; all other checked-in OCID-VLG configs use
the compact RGB-only contract.

The adapter keeps original-coordinate grasp rectangles for Jacquard evaluation,
then transforms the rectangle corners before generating input-resolution grasp
maps. This avoids the fixed-416 map misalignment in the legacy loader. It also
supports the center-offset supervision required by CROG-OFF and DROG-OFF.

Inspect one expression before training:

```bash
python tools/inspect_ocid_vlg_sample.py \
  --dataset-root /path/to/OCID-VLG \
  --version multiple --split train --index 0
```

## ETRG-A RGB-D on Ascend

The `etrg` registry entry integrates the official ETRG-A model with explicit
OCID-VLG depth routing through the NPU device layer. R50 and R101 configs are
provided under `config/ocid_vlg/`; VCoT and Grasp-Tools are intentionally not
configured because they do not provide aligned sensor depth.

Verify the NPU environment, local weights, and one random RGB-D forward pass:

```bash
python tools/check_npu_env.py \
  --config configs/etrg/etrg_r50_ocid_vlg.yaml --forward
```

Then run two NPUs:

```bash
torchrun --nproc_per_node=2 tools/train.py \
  --config configs/etrg/etrg_r50_ocid_vlg.yaml --opts \
  DATA.root_path /mnt/ssd0/mengyuan/data/OCID-VLG
```

See [docs/etrg.md](docs/etrg.md) for dataset, weight, checkpoint, and Ascend
compatibility details.

## MapleGrasp on Ascend

`model/maplegrasp.py` ports the CROG-based mask-guided projector into the NPU
model registry. A detached predicted segmentation mask gates the four grasp
branches before the language-conditioned dynamic convolution. The model uses
device-agnostic PyTorch operators and relies on `NPUGraspRunner` for placement;
evaluation never consumes a ground-truth object mask.

Run joint training on two NPUs with:

```bash
torchrun --nproc_per_node=2 train.py \
  --config config/ocid_vlg/maplegrasp.yaml --opts \
  DATA.root_path /path/to/OCID-VLG
```

For the paper-style two-stage schedule, first use
`TRAIN.maple_stage segmentation`, then initialize a second run with
`TRAIN.maple_stage grasp` and `TRAIN.weight` pointing at the stage-one
`best_iou_model.pth`. Grasp-Tools and VCoT configs are provided in their
respective directories.

Train any supported grasp model with its OCID-VLG config, for example:

```bash
python train.py --config config/ocid_vlg/drog.yaml --opts \
  DATA.root_path /path/to/OCID-VLG
```

## Training

Single NPU:

```bash
python train.py --config config/grasp_tools/drogoff.yaml
```

Distributed:

```bash
torchrun --nproc_per_node=2 train.py \
  --config config/grasp_tools/drogoff.yaml
```

Evaluation:

```bash
python evaluate.py \
  --config config/grasp_tools/drogoff.yaml \
  --checkpoint exp/grasp_tools/drogoff_grasp_tools/best_jindex_model.pth
```

## Output contract

Grasp-aware models return segmentation, quality, sine, cosine, and width maps.
Offset variants append a `(dx, dy)` map normalized by `DATA.offset_r`.
GGCNN-CLIP and GR-ConvNet-CLIP are grasp-only baselines, so their quality map
also occupies the segmentation slot required by the shared engine.

`LGD` is a ToolRGS dense-map port of Language-driven Grasp Detection. It keeps
the public cosine diffusion schedule, x0 quality-map denoising, language/image
conditioning, and contrastive alignment while exposing the shared segmentation,
quality, sine, cosine, and width contract. `TRAIN.lgd_sampling_steps` controls
the DDIM inference cost; use `1000` for the full training schedule or a smaller
value for faster comparison. The upstream LGD MIT notice is in
`model/lgd_LICENSE`. See the
[CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Vuong_Language-driven_Grasp_Detection_CVPR_2024_paper.html)
and [official implementation](https://github.com/Fsoft-AIC/LGD).

`GraspMamba` is a ToolRGS paper reimplementation, not the unreleased official
training code. It follows the paper's four-stage MambaVision backbone, frozen
CLIP text encoder, per-stage visual-language fusion, and recursive top-down
feature aggregation. The adapter adds an instance-segmentation head and emits
the shared dense grasp maps required by the ToolRGS engine. VCoT/Grasp-Anything
is the paper-aligned training dataset; the Grasp-Tools and OCID-VLG configs are
cross-dataset compatibility experiments rather than paper-reported settings.
See the [GraspMamba paper](https://arxiv.org/abs/2409.14403) and the
[official MambaVision backbone](https://github.com/NVlabs/MambaVision).

Run the paper-aligned experiment with:

```bash
python train.py --config config/vcot/graspmamba.yaml --opts \
  DATA.root_path /mnt/ssd0/mengyuan/data/grasp-anything
```

## Environment

Install CANN and the mutually compatible `torch`/`torch_npu` wheels first, then
install `requirement-npu.txt`. The exact versions are determined by the CANN
release on the Ascend server; do not blindly install the old CUDA project's
PyTorch pin. Pretrained CLIP and DINOv2 weights are not stored in Git.

Check the NPU runtime before downloading datasets or starting a long job:

```bash
python tools/check_npu_env.py
python tools/check_npu_env.py --config config/vcot/drogoff.yaml --forward
```

GraspMamba is the one explicit compatibility boundary: upstream MambaVision
loads the CUDA-only `selective_scan_cuda` extension. It remains available for
future Ascend selective-scan integration, but is not claimed NPU-ready. The
other nine grasp architectures use the explicit NPU runtime path.

The configured official MambaVision checkpoint is downloaded automatically if
it is missing and the server has network access. Otherwise download it once and
set `TRAIN.mamba_pretrain` to the local file. MambaVision code uses NVIDIA's
non-commercial source license and its pretrained weights use CC-BY-NC-SA-4.0;
check those terms before redistribution or commercial use.

## Real-world demo and robot sender

ToolRGSNPU includes a configuration-driven PyQt demo ported from the local server
CROG deployment. It supports every registered ToolRGS grasp architecture, OpenCV/video,
RealSense, GStreamer shared memory, optional MMDetection and Whisper, and the
legacy Kinova TCP command format. Start in dry-run mode:

```bash
cp config/deployment/lab.example.yaml config/deployment/lab.yaml
python tools/check_deployment.py --config config/deployment/lab.yaml \
  --probe-camera --build-model
python deploy_gui.py --config config/deployment/lab.yaml
```

See [docs/real_world_deployment.md](docs/real_world_deployment.md) before
enabling robot output. The repository contains the sender but not the external
Kinova receiver/controller or its calibration, so a clone alone cannot safely
move the physical robot.

## Component architecture

ToolRGS now has MMDetection-style registries for models, datasets, transforms,
losses, metrics, postprocessors, loops, hooks, cameras, robot clients,
detectors, and audio inputs.
Existing training configs and builders remain compatible while new components
can be selected by `type` without extending central `if/elif` factories.

```bash
python tools/list_components.py
```

Dense model tuples can be normalized into named `GraspOutput`, `GraspTargets`,
and `GraspModelResult` structures. The main paths now use registered
`GraspTrainLoop` and `GraspValLoop` components; validation and deployment share
the `DenseGraspPostProcessor` decoding contract. See
[docs/component_architecture.md](docs/component_architecture.md) for extension
examples and the compatibility plan.

## Acknowledgements

ToolRGS integrates ideas and code from CROG, MapleGrasp, DETRIS, DINOv2, and CRIS. Preserve
their citations and licenses when publishing derived results.
