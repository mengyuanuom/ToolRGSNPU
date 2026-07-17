# Grasp-Tools v2 compositional dataset

ToolRGS includes the full annotated Grasp-Tools source set and all background
images needed to generate the v2 dataset. A fresh clone therefore does not need
a separate download before augmentation.

## Included source data

```text
assets/grasp_tools/
├── graspall/       # 107 JPG + 107 JSON annotations
└── backgrounds/    # 42 background JPG images
```

The generator validates the 22 canonical categories and skips the two empty
object records in `000000000076.json` with explicit warnings.

## Smoke test

From the ToolRGS repository root:

```bash
python -u tools/dataset_converters/grasp_tools/augment.py \
  --out-dir /tmp/grasp_tools_v2_smoke \
  --smoke-test \
  --image-ext jpg \
  --overwrite
```

Inspect `/tmp/grasp_tools_v2_smoke/_preview` before starting the full run.

## Full generation

The repository's `datasets` symlink normally points to `../data`. The default
command therefore writes the generated corpus outside Git while keeping all
source assets inside the clone:

```bash
python -u tools/dataset_converters/grasp_tools/augment.py \
  --train-scenes 3000 \
  --val-scenes 500 \
  --test-scenes 1000 \
  --objects-min 3 \
  --objects-max 5 \
  --queries-min 4 \
  --queries-max 8 \
  --scales 0.6,0.8,1.0,1.25,1.5 \
  --angle-bins 12 \
  --same-category-probability 0.40 \
  --hard-negative-probability 0.30 \
  --grasp-height 20 \
  --image-ext jpg \
  --jpeg-quality 95
```

The output defaults to `datasets/grasp-tools/aug_graspall_v2` and contains
`train`, `val`, `test`, `_preview`, `metadata.json`, and one `index.jsonl` per
split.

Each scene image is stored once. Its JSON contains multiple language queries;
every query records `text`, `target_idx`, `type`, `difficulty`, and a symbolic
program. `objects[target_idx]` supplies the target mask and grasp rectangles.

## Train DROG-OFF

The v2-aware `GraspToolDataset` expands those lightweight queries without
duplicating scene images in memory or on disk. The supplied config also raises
the CLIP token limit to 32 for relational descriptions:

```bash
python train.py --config config/grasp_tools/drogoff_v2.yaml
```

For two NPUs:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --config config/grasp_tools/drogoff_v2.yaml
```

The synthetic splits use disjoint background files and held-out evaluation
language templates, but share the same 107 source cutouts. They measure
compositional and language generalization, not novel physical instances.
Use the separate physical-scene protocol for real-world generalization claims.
