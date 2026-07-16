# ETRG-A integration

ToolRGSNPU includes the official ETRG-A RGB-D model blocks under `model/etrg/`
and retains their BSD-2-Clause notice. The integration adds ToolRGS structured
outputs, registry construction, explicit depth routing, preflight weight
checks, and the shared evaluator.

Official ETRG-A parameter names are retained so released R50/R101 checkpoints
can be supplied through `TRAIN.weight` for initialization or evaluated through
`evaluate.py`.

## OCID-VLG download and directory

Download `OCID-VLG.zip` from the
[official dataset repository](https://github.com/gtziafas/OCID-VLG) or the
[official Google Drive file](https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link).
The archive contains the RGB scenes, aligned depth, instance masks, grasp
annotations, and language-expression splits needed by ETRG.

Do not use the smaller `OCID-VLG-S.zip` ToolRGSNPU Release for ETRG-A: that
RGB-only subset intentionally removes every depth and PCD directory.

Recommended server layout:

```text
/mnt/ssd0/mengyuan/
├── ToolRGSNPU/
│   └── datasets -> ../data
└── data/
    └── OCID-VLG/
        ├── refer/multiple/
        │   ├── train_expressions.json
        │   ├── val_expressions.json
        │   └── test_expressions.json
        └── <sequence>/
            ├── rgb/<image_name>
            ├── depth/<image_name>
            └── seg_mask_instances_combi/<image_name>
```

Download and extract on the server:

```bash
python -m pip install gdown
mkdir -p /mnt/ssd0/mengyuan/data
cd /mnt/ssd0/mengyuan/data
gdown --fuzzy \
  'https://drive.google.com/file/d/1VwcjgyzpKTaczovjPNAHjh-1YvWz9Vmt/view?usp=share_link' \
  -O OCID-VLG.zip
unzip OCID-VLG.zip
cd /mnt/ssd0/mengyuan/ToolRGSNPU
ln -sfn ../data datasets
```

The local `datasets` symlink is ignored by Git and must be created once per
checkout. The configured root must be the directory directly containing
`refer/`. Check
the result before training:

```bash
cd /mnt/ssd0/mengyuan/ToolRGSNPU
readlink -f datasets
test -f datasets/OCID-VLG/refer/multiple/train_expressions.json
python tools/inspect_ocid_vlg_sample.py \
  --dataset-root datasets/OCID-VLG --version multiple --split train --index 0
```

All supplied OCID-VLG configs default to `./datasets/OCID-VLG`. For a different
location, keep the files outside Git and override the path without editing YAML:

```bash
python tools/train.py --config configs/etrg/etrg_r50_ocid_vlg.yaml --opts \
  DATA.root_path /absolute/path/to/OCID-VLG
```

## Supported experiment

The paper-aligned path is OCID-VLG with aligned RGB and depth images:

```bash
torchrun --nproc_per_node=2 tools/train.py \
  --config configs/etrg/etrg_r50_ocid_vlg.yaml --opts \
  DATA.root_path /path/to/OCID-VLG \
  TRAIN.clip_pretrain pretrain/RN50.pt
```

`configs/etrg/etrg_r101_ocid_vlg.yaml` selects the stronger CLIP RN101 variant. Batch
sizes are per NPU. The supplied values are conservative Ascend starting points,
not published throughput settings; tune them after a successful smoke test.

ETRG requires `DATA.with_depth: true`. ToolRGSNPU fails before dataset construction
if depth is disabled and fails at the first malformed batch if `depth` is
missing. VCoT and Grasp-Tools currently expose placeholder zero depth rather
than aligned sensor depth, so no ETRG configs are supplied for those datasets.

## Weights

CLIP weights are local files and are checked before model construction:

- R50: `pretrain/RN50.pt`
- R101: `pretrain/RN101.pt`

Download either CLIP file and the optional ResNet-18 file with the commands in
the [pretrained-weight guide](pretrained_weights.md); it also contains the
official direct URLs.

The ResNet-18 depth encoder uses torchvision ImageNet weights by default and
may download them into the PyTorch cache. On an offline server, download a
torchvision ResNet-18 state dict separately and set:

```yaml
TRAIN:
  depth_pretrain: /absolute/path/to/resnet18-f37072fd.pth
```

Set `depth_backbone_pretrained: false` only when intentionally training the
depth encoder from scratch.

When `TRAIN.weight` or `TRAIN.resume` points to a full ETRG checkpoint, the
separate torchvision download is skipped because that checkpoint restores the
depth encoder too.

Install a torchvision build compatible with the selected PyTorch/torch_npu
pair. Verify model construction and the RGB-D forward path before training:

```bash
python tools/check_npu_env.py \
  --config configs/etrg/etrg_r50_ocid_vlg.yaml --forward
```

The NPU port replaces upstream `tensor_split` with `torch.chunk` for a more
portable Ascend graph. If an installed CANN/torch_npu combination reports an
unsupported attention operator, rerun this smoke test with `TRAIN.amp False`
to identify the exact dtype/operator boundary.

## Upstream

- Paper: https://arxiv.org/abs/2409.19457
- Official ETRG-A code: https://github.com/hjy-u/ETRG-RGS

The upstream result uses a dedicated depth preprocessing path. ToolRGSNPU matches
its inverse per-sample maximum normalization in the model so other registered
models continue to receive the unchanged dataset contract.
