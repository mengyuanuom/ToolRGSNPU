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

## Full difficulty-4 generation

The repository's `datasets` symlink normally points to `../data`. The default
command therefore writes the generated corpus outside Git while keeping all
source assets inside the clone:

```bash
python -u tools/dataset_converters/grasp_tools/augment.py \
  --out-dir datasets/grasp-tools/aug_graspall_v2_full \
  --train-scenes 3000 \
  --val-scenes 500 \
  --test-scenes 1000 \
  --objects-min 3 \
  --objects-max 5 \
  --queries-min 4 \
  --queries-max 8 \
  --max-query-difficulty 4 \
  --language-templates heldout \
  --category-vocabulary canonical \
  --scales 0.6,0.8,1.0,1.25,1.5 \
  --angle-bins 12 \
  --same-category-probability 0.40 \
  --hard-negative-probability 0.30 \
  --brightness-jitter 0.12 \
  --contrast-jitter 0.12 \
  --saturation-jitter 0.10 \
  --grasp-height 20 \
  --image-ext jpg \
  --jpeg-quality 95
```

This command writes `datasets/grasp-tools/aug_graspall_v2_full` and contains
`train`, `val`, `test`, `_preview`, `metadata.json`, and one `index.jsonl` per
split.

## Starter dataset: category queries only

For the simplest first-stage experiment, place two or three larger tools on
varied backgrounds, keep every category unique within a scene, and create one
category query for each target object. Train, validation, and test share one
larger prompt pool while retaining independent rendered scenes and disjoint
background-image splits. This starter protocol is now the generator default,
so a full dataset only needs one command from the repository root:

```bash
python -u tools/dataset_converters/grasp_tools/augment.py
```

The expanded form below is equivalent and remains useful when overriding an
individual setting:

```bash
python -u tools/dataset_converters/grasp_tools/augment.py \
  --out-dir datasets/grasp-tools/aug_graspall_v2 \
  --train-scenes 3000 \
  --val-scenes 500 \
  --test-scenes 1000 \
  --objects-min 2 \
  --objects-max 3 \
  --queries-min 2 \
  --queries-max 4 \
  --max-query-difficulty 1 \
  --language-templates shared \
  --category-vocabulary expanded \
  --scales 0.9,1.0,1.15,1.3 \
  --angle-bins 8 \
  --same-category-probability 0 \
  --hard-negative-probability 0 \
  --brightness-jitter 0.05 \
  --contrast-jitter 0.05 \
  --saturation-jitter 0.05 \
  --grasp-height 20 \
  --image-ext jpg \
  --jpeg-quality 95
```

The starter prompt pool combines 22 short grasp instructions with four terms
for each of the 22 canonical classes. Canonical labels remain unchanged, while
expanded category queries carry `prompt_cycle: category_v1`. During training,
each target receives its own reproducibly shuffled 88-prompt order derived
from `dynamic_prompt_seed + scene_id + target_idx`; any run shorter than 88
epochs simply consumes the corresponding unique prefix. Validation and test
always retain their fixed JSON text. Set `DATA.dynamic_train_prompts` to
`False` for a fixed-prompt ablation, or change `DATA.dynamic_prompt_seed` to
compare language schedules.

Difficulty levels are cumulative: `1` keeps category queries, `2` adds
absolute-location queries, `3` adds same-category and single-reference spatial
queries, and `4` also enables between-object relations. The default generator
settings use the difficulty-1 starter protocol; pass the full set of options
above when running the difficulty-4 comparison.

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
