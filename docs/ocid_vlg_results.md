# OCID-VLG test results

This page records the selected OCID-VLG test-set comparison for ToolRGSNPU.
It deliberately mixes two result sources only where requested:

- **Project evaluation**: checkpoints evaluated by this project on the
  17,749-sample OCID-VLG `test` split with the `crog_legacy` protocol.
- **Paper**: the number printed in the original CROG or MapleGrasp paper.

The source is shown for every row. Paper numbers are not silently replaced by
our checkpoint re-evaluations.

## Selected comparison

| Model | Result source | IoU | Pr@50 | Pr@60 | Pr@70 | Pr@80 | Pr@90 | J@1 | J@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DrogOff (ours)** | Project evaluation | **81.56** | **97.24** | **96.08** | **89.83** | **70.21** | **23.35** | **88.15** | **93.09** |
| CROG | Original CROG paper | 81.10 | 96.90 | 94.80 | 87.20 | 64.10 | 16.40 | 77.20 | 87.70 |
| GRConvNetCLIP | Project evaluation | N/A | N/A | N/A | N/A | N/A | N/A | 88.14 | 91.21 |
| LGD | Project evaluation, training-era compatible code | 65.94 | 88.38 | 79.14 | 54.71 | 26.96 | 0.02 | 84.94 | 87.27 |
| ETRG | Project evaluation | 74.97 | 90.21 | 87.27 | 78.70 | 53.71 | 12.66 | 73.73 | 76.85 |
| GGCNNCLIP | Project evaluation, training-era compatible code | N/A | N/A | N/A | N/A | N/A | N/A | 15.32 | 17.13 |
| MapleGrasp-CROG | Original MapleGrasp paper | 81.36 | 97.40 | 95.32 | 87.90 | 65.40 | 16.40 | 86.15 | **91.90** |

The original CROG and MapleGrasp tables label their top-5 grasp-success column
as `J@Any`. This comparison places those paper values in `J@5` because the
benchmark evaluation uses the same top-5 candidate standard. `J@1` uses the
standard Jacquard criterion: grasp-rectangle IoU greater than 0.25 and an angle
difference below 30 degrees.

GRConvNetCLIP and GGCNNCLIP do not have a referring-segmentation head. Their
wrappers reuse the grasp-quality map as `ins_pred`, so segmentation IoU and
Precision@X are marked N/A instead of presenting those values as meaningful
segmentation results.

## Checkpoint provenance for project evaluations

| Model | Checkpoint | Evaluation note |
| --- | --- | --- |
| DrogOff | `best_iou_epoch_046.pth` | Independent test/retest in `ToolRGSNPU`; both runs reproduce IoU 81.56, J@1 88.15, and J@5 93.09. |
| GRConvNetCLIP | `best_epoch_036_J1_86.27_J5_90.94.pth` | Valid current-code clamp evaluation. |
| LGD | `best_epoch_035_J1_84.28_J5_88.71.pth` | Evaluated with training-era commit `b09d9bf`; later forward-semantic changes are incompatible with this checkpoint. |
| ETRG | `best_epoch_034_J1_73.55_J5_78.77.pth` | Valid current-code clamp evaluation. |
| GGCNNCLIP | `best_epoch_036_J1_20.26_J5_24.32.pth` | Evaluated with training-era commit `b09d9bf`; later FiLM/text-normalization semantics are incompatible with this checkpoint. |

The earlier archived DrogOff run (`best_epoch_030_J1_91.24_J5_94.12.pth`)
reached IoU 81.23, J@1 89.30, and J@5 92.94. The table uses the independently
retested current ToolRGSNPU checkpoint above as the primary project result and
retains this historical result here to avoid hiding a stronger archived J@1.

## Paper-number provenance

- **CROG**: Table 2 of *Language-guided Robot Grasping: CLIP-based Referring
  Grasp Synthesis in Clutter* reports IoU 81.1, J@1 77.2, and J@Any/J@5 87.7 on
  the OCID-VLG test split. Paper: <https://proceedings.mlr.press/v229/tziafas23a.html>.
- **MapleGrasp**: Table 2 of *MapleGrasp: Mask-guided Feature Pooling for
  Language-driven Efficient Robotic Grasping* reports the selected
  MapleGrasp-CROG variant above. This is the paper's stronger grasping variant.
  The alternative MapleGrasp-Ref2Grab variant reports IoU 83.78, J@1 76.8,
  and J@Any/J@5 84.7. Paper: <https://arxiv.org/abs/2506.06535>.

## Reading the table

- DrogOff has the strongest selected J@1 result at 88.15.
- DrogOff and MapleGrasp-CROG have similar reported segmentation IoU, 81.56
  and 81.36 respectively, but they come from different result sources.
- DrogOff also leads the aligned top-5 grasp-success comparison at 93.09,
  followed by MapleGrasp-CROG at 91.90.
