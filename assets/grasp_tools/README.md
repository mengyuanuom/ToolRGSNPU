# Embedded Grasp-Tools source assets

This directory contains the complete source material required by ToolRGS's
compositional Grasp-Tools augmentation pipeline:

- `graspall_v3/`: the active V3 release with 107 RGB images and 107 reviewed
  polygon/grasp JSON annotations (107 objects, 7,820 grasp rectangles);
- `graspall/`: the archived V2 source release retained for reproducibility;
- `backgrounds/`: 42 tool-free background images.

Both releases use the same 22 canonical categories and the same source RGB
images. V3 replaces every JSON sidecar with the reviewed annotation set and
contains no empty object records. The augmentation generator defaults to V3.

Do not commit generated `aug_graspall_v3_15k` scenes here. Generate them below the
normal `datasets/grasp-tools/` data root by following
`docs/grasp_tools_v3.md`.
