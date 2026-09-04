# Grasp-Tools V3 source data and augmentation

ToolRGSNPU embeds the complete reviewed Grasp-Tools V3 source release so a
fresh clone can reproduce the unified V3 training data without downloading a
separate annotation archive.

## Included assets

```text
assets/grasp_tools/
├── graspall_v3/    # 107 JPG images + 107 reviewed JSON annotations
├── graspall/       # archived V2 source release
└── backgrounds/    # 42 tool-free background images
```

V3 contains 107 valid objects across 22 canonical categories and 7,820 grasp
rectangles. Every image has one same-stem JSON sidecar. The RGB images are the
same photographs as V2; the V3 release replaces the annotations and removes
the two empty provenance records present in the V2 sidecars.

## Generate the augmented V3 dataset

From the repository root, run:

```bash
python -u tools/dataset_converters/grasp_tools/augment.py
```

The defaults read `assets/grasp_tools/graspall_v3` and write
`datasets/grasp-tools/aug_graspall_v3_15k`. The generated dataset is intentionally
ignored by Git because it is reproducible from the embedded source assets and
generator.

Run a small end-to-end validation first when moving to a new environment:

```bash
python -u tools/dataset_converters/grasp_tools/augment.py \
  --out-dir /tmp/grasp_tools_v3_smoke \
  --smoke-test \
  --overwrite
```

The generator keeps the RGB cutout, polygon mask, bounding box, and every grasp
rectangle under the same affine transform. Its 12,000/1,000/2,000 scene split,
2--3 objects per scene, 2--4 queries per scene, scale set, rotation bins,
language settings, and seed match the current ToolRGS V3 generation contract.
The default training profiles under `config/grasp_tools/` all point to
`./datasets/grasp-tools/aug_graspall_v3_15k`.

For the complete generator options, schema, curriculum levels, and balance
contract, see [the detailed compositional-data guide](grasp_tools_v2.md). Its
algorithm description still applies; use the V3 source/output paths above.
