# Full benchmark result archive: OCID-VLG, Grasp-Tools V3, VCoT-GraspSet, and GraspNet-VLG

> Internal analysis snapshot, 2026-09-12. For the compact publication-facing
> table, see the [concise paper comparison](ocid_vlg_results.md).

This archive keeps all currently consolidated metrics, result-source labels,
checkpoint/epoch provenance, and internal GraspNet-VLG variants. Paper numbers
and project evaluations remain explicitly distinguished; rows with unavailable
complete evaluations stay blank rather than being inferred. The detailed result
snapshot for OCID-VLG, VCoT-GraspSet, and GraspNet-VLG is preserved from
repository commit 69f1e85, while the Grasp-Tools V3 section is preserved from
commit 8e97d3f; only document structure and ranking emphasis were changed.

The OCID-VLG table additionally includes the eight-NPU GraspMamba and MapleGrasp evaluations completed on 2026-09-12 after synchronizing the codebase to `bbf943d`.

## OCID-VLG

This page records the selected OCID-VLG test-set comparison for ToolRGSNPU.
It deliberately mixes two result sources only where requested:

- **Project evaluation**: checkpoints evaluated by this project on the
  17,749-sample OCID-VLG `test` split with the `crog_legacy` protocol.
- **Paper**: the number printed in the original CROG or MapleGrasp paper.

The source is shown for every row. Paper numbers are not silently replaced by
our checkpoint re-evaluations.

### Selected comparison

| Model | Result source | IoU | Pr@50 | Pr@60 | Pr@70 | Pr@80 | Pr@90 | J@1 | J@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **DrogOff (ours)** | Project evaluation | **81.56** | **97.24** | **96.08** | **89.83** | **70.21** | **23.35** | **88.15** | **93.09** |
| **MapleGrasp Stage 2 (ours)** | Project evaluation | 80.54 | **97.28** | 94.97 | 88.24 | 65.86 | 19.81 | **88.16** | 92.36 |
| CROG | Original CROG paper | 81.10 | 96.90 | 94.80 | 87.20 | 64.10 | 16.40 | 77.20 | 87.70 |
| GRConvNetCLIP | Project evaluation | N/A | N/A | N/A | N/A | N/A | N/A | 88.14 | 91.21 |
| GraspMamba | Project evaluation | 76.39 | 90.50 | 88.59 | 83.45 | 61.24 | 18.22 | 85.86 | 88.44 |
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

### Checkpoint provenance for project evaluations

| Model | Checkpoint | Evaluation note |
| --- | --- | --- |
| DrogOff | `best_iou_epoch_046.pth` | Independent test/retest in `ToolRGSNPU`; both runs reproduce IoU 81.56, J@1 88.15, and J@5 93.09. |
| MapleGrasp Stage 2 | `best_j1_epoch_045.pth` | Validation-J@1-selected checkpoint; eight-NPU test evaluation, global batch 8 / per-rank batch 1: IoU 80.54, J@1 88.16, J@5 92.36. |
| GRConvNetCLIP | `best_epoch_036_J1_86.27_J5_90.94.pth` | Valid current-code clamp evaluation. |
| LGD | `best_epoch_035_J1_84.28_J5_88.71.pth` | Evaluated with training-era commit `b09d9bf`; later forward-semantic changes are incompatible with this checkpoint. |
| GraspMamba | `best_j1_epoch_047.pth` | Validation-J@1-selected checkpoint; eight-NPU test evaluation, global batch 8 / per-rank batch 1: IoU 76.39, J@1 85.86, J@5 88.44. |
| ETRG | `best_epoch_034_J1_73.55_J5_78.77.pth` | Valid current-code clamp evaluation. |
| GGCNNCLIP | `best_epoch_036_J1_20.26_J5_24.32.pth` | Evaluated with training-era commit `b09d9bf`; later FiLM/text-normalization semantics are incompatible with this checkpoint. |

The earlier archived DrogOff run (`best_epoch_030_J1_91.24_J5_94.12.pth`)
reached IoU 81.23, J@1 89.30, and J@5 92.94. The table uses the independently
retested current ToolRGSNPU checkpoint above as the primary project result and
retains this historical result here to avoid hiding a stronger archived J@1.

### Paper-number provenance

- **CROG**: Table 2 of *Language-guided Robot Grasping: CLIP-based Referring
  Grasp Synthesis in Clutter* reports IoU 81.1, J@1 77.2, and J@Any/J@5 87.7 on
  the OCID-VLG test split. Paper: <https://proceedings.mlr.press/v229/tziafas23a.html>.
- **MapleGrasp**: Table 2 of *MapleGrasp: Mask-guided Feature Pooling for
  Language-driven Efficient Robotic Grasping* reports the selected
  MapleGrasp-CROG variant above. This is the paper's stronger grasping variant.
  The alternative MapleGrasp-Ref2Grab variant reports IoU 83.78, J@1 76.8,
  and J@Any/J@5 84.7. Paper: <https://arxiv.org/abs/2506.06535>.

Both new project rows use checkpoints selected only by validation J@1. Test evaluation covered all 17,749 OCID-VLG test samples with eight ranks, global batch 8 and per-rank batch 1; test metrics were not used for checkpoint selection.

### Reading the table

- MapleGrasp Stage 2 has the strongest displayed project J@1 at 88.16, effectively tied with DrogOff at 88.15 and GRConvNetCLIP at 88.14.
- DrogOff retains the strongest project segmentation IoU (81.56) and J@5 (93.09); MapleGrasp Stage 2 reaches 80.54 IoU and 92.36 J@5.
- GraspMamba reaches 76.39 IoU, 85.86 J@1 and 88.44 J@5.
- The locally evaluated MapleGrasp Stage 2 row is distinct from the paper-reported MapleGrasp-CROG row, which is not a directly reproduced checkpoint in this repository.

## Grasp-Tools V3

Grasp-Tools V3 reports referring-segmentation IoU separately from grasp
success. The grasp comparison uses the top-1 decoded grasp for every sample
and evaluates it against all valid ground-truth grasps for that sample.

For a prediction `p` and a ground-truth grasp `g`, the parallel-jaw angle
error is 180-degree periodic:

```text
angle_error(p, g) = abs(((theta_p - theta_g + 90) mod 180) - 90)
```

For an IoU threshold `t` and angle threshold `a`, a sample succeeds when at
least one ground-truth grasp has continuous rotated-rectangle IoU greater
than `t` and angle error at most `a`. Its success rate is:

```text
SR(t, a) = 100 / N * sum(success_i(t, a))
```

The benchmark evaluates the Cartesian product of IoU thresholds
`{0.25, 0.50, 0.75}` and angle thresholds `{5, 10, 20, 30}` degrees. The
multi-threshold mean Success Rate is the unweighted mean of all 12 cells:

```text
mSR = 1 / 12 * sum_t sum_a SR(t, a)
```

Every sample and every threshold pair therefore has equal weight. Unlike the
traditional permissive `SR(0.25, 30 deg)` score, mSR also penalizes inaccurate
grasp overlap and angle. Grasp-size decoding must match training: the selected
DrogOff V1 checkpoint uses sigmoid decoding, while CROG uses clamp decoding.

### Selected comparison

All values are percentages on the same Grasp-Tools V3 validation split using
each model's selected checkpoint. A dash means that no complete aligned V3
evaluation is currently available.

| Model | Result source | Size decoder | Seg. IoU | mSR | SR(0.25, 30 deg) | SR(0.25, 10 deg) | SR(0.50, 10 deg) | SR(0.75, 30 deg) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **DrogOff V1 (ours)** | Project evaluation | sigmoid | **83.78** | **89.14** | **99.51** | **97.45** | **96.52** | **75.67** |
| CROG | Project evaluation | clamp | 81.72 | 75.28 | 99.15 | 95.75 | 90.81 | 43.77 |
| GRConvNetCLIP | - | - | - | - | - | - | - | - |
| LGD | - | - | - | - | - | - | - | - |
| ETRG | - | - | - | - | - | - | - | - |
| GGCNNCLIP | - | - | - | - | - | - | - | - |
| MapleGrasp-CROG | - | - | - | - | - | - | - | - |

### Checkpoint provenance for project evaluations

| Model | Selected checkpoint | Evaluation contract |
| --- | --- | --- |
| DrogOff V1 | `best_msr_model.pth` (epoch 24) | Sigmoid grasp-size decoding. |
| CROG | `best_msr_model.pth` (epoch 36) | Clamp grasp-size decoding. |

DrogOff V1 leads CROG by 13.86 mSR points. Their permissive
`SR(0.25, 30 deg)` scores are both above 99%, but the gap widens under strict
overlap: `SR(0.75, 30 deg)` is 75.67 for DrogOff V1 and 43.77 for CROG.

The same CROG checkpoint previously produced 42.05 mSR when it was decoded
with sigmoid by mistake. The aligned clamp evaluation produces the 75.28 mSR
reported above; results from mismatched train/inference decoding are excluded.

## VCoT-GraspSet

VCoT-GraspSet is a language-conditioned planar-grasp benchmark refined from
Grasp Anything. Each sample contains a 416 x 416 RGB scene, a language
instruction identifying the target object, and one or more ground-truth grasp
rectangles. The benchmark covers 388 object categories: 367 seen categories
contribute more than 186K training samples and a 3,000-sample seen test split,
while 21 held-out categories form the 1,487-sample unseen test split.
ToolRGSNPU reserves the final 5,000 rows of the supplied training CSV for
checkpoint validation.

Evaluation emits exactly one grasp rectangle per sample. A prediction is
successful when its rotated IoU with at least one ground-truth grasp is at
least 0.25 and its 180-degree-periodic angle error is at most 30 degrees.
`Seen` and `Unseen` below are grasp success rates. Following the VCoT-Grasp
paper, `Avg.` is their harmonic mean:

```text
Avg. = 2 * Seen * Unseen / (Seen + Unseen)
```

### Selected comparison

| Model | Seen | Unseen | Avg. |
| --- | ---: | ---: | ---: |
| **DrogOff V2** | **93.87** | **59.18** | **72.59** |
| DROG | 90.57 | 57.36 | 70.24 |
| DrogOff V1 | 89.83 | 53.67 | 67.19 |
| CROG-OFF | 88.73 | 47.61 | 61.97 |
| CROG | 86.37 | 43.44 | 57.81 |
| LGD (VCoT-Grasp paper) | 38.67 | 13.42 | 19.93 |
| GR-ConvNet + CLIP (VCoT-Grasp paper) | 70.80 | 33.29 | 45.29 |
| GG-CNN + CLIP (VCoT-Grasp paper) | 56.33 | 17.89 | 27.16 |
| ETRG | - | - | - |
| MapleGrasp | - | - | - |
| VCoT-Grasp, MLP head (VCoT-Grasp paper) | 73.37 | 52.25 | 61.03 |
| VCoT-Grasp, LM head (VCoT-Grasp paper) | 83.60 | 58.98 | 69.16 |

The CROG/DROG-family rows are aligned ToolRGSNPU evaluations using the same
official seen and unseen splits. Rows explicitly marked `VCoT-Grasp paper`
are numbers from Table III of the original paper. A dash means that no
complete, valid local seen/unseen test pair is currently available; partial
or interrupted training runs are not promoted into the comparison.

### Checkpoint provenance for project evaluations

| Model | Selected checkpoint |
| --- | --- |
| CROG | `best_j1_epoch_012.pth` |
| CROG-OFF | `best_j1_epoch_009.pth` |
| DROG | `best_j1_epoch_028.pth` |
| DrogOff V1 | `best_j1_epoch_008.pth` |
| DrogOff V2 | `best_j1_epoch_036.pth` |

Earlier VCoT checkpoints produced by the legacy CROG-NPU training profiles
are retained as historical records but are not mixed into this table. Those
profiles differ in grasp-size scaling, coordinates, target selection, adapter
semantics, or decoding, so the aligned ToolRGSNPU evaluations supersede them
for the primary comparison.

### Paper-number provenance and interpretation

The VCoT-Grasp paper values come from Table III of
*VCoT-Grasp: Grasp Foundation Models with Visual Chain-of-Thought Reasoning
for Language-driven Grasp Generation*:
<https://arxiv.org/html/2510.05827v1#S4.SS1>.

DrogOff V2 has the strongest selected result in the combined table: 93.87 on
Seen, 59.18 on Unseen, and a 72.59 harmonic mean. Numerically, its Unseen
score is 0.20 points above the paper's VCoT-Grasp LM-head result. This is a
same-split, same-success-criterion score comparison rather than an
equal-training-budget comparison: the paper trains its 224-pixel models for
three epochs, whereas the aligned ToolRGSNPU models use 448-pixel inputs and
longer schedules.

## GraspNet-VLG / RealVLG

GraspNet-VLG is the executable GraspNet subset of the public RealVLG-11B
benchmark. Training uses scenes 0000-0099. Official testing evaluates every
object with nonempty grasps in frame `0000.json` from three held-out scene
ranges: Seen uses scenes 0100-0129 (253 samples), Similar uses 0130-0159
(235 samples), and Novel uses 0160-0189 (164 samples).

Segmentation reports generalized IoU (`gIoU`), cumulative IoU (`cIoU`),
`F_beta`, `S_alpha`, `E_measure`, and the segmentation validity rate. Grasp
evaluation emits one rectangle with a fixed 40-pixel gripper depth. `gAcc`
requires best rotated IoU greater than 0.25 and angle error below 30 degrees;
`Grasp mIoU` is the mean best grasp overlap over valid predictions.

All values below are percentages. Every completed row uses the experiment's
`best_iou` checkpoint and an independent evaluation on all three official
splits. No selective ranking emphasis is applied in this archive. DrogOff Native V3
LoRA and DrogOff Native V4 are retained here as internal analysis variants even
though they are intentionally omitted from the concise publication table.

### Selected comparison

| Model (best-IoU epoch) | Split | gIoU | cIoU | F_beta | S_alpha | E_measure | Seg. valid | Grasp mIoU | Grasp valid | gAcc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DrogOff Offset V2 (17) | Seen | 65.03 | 65.34 | 82.77 | 45.13 | 98.17 | 100.00 | 44.62 | 100.00 | 75.49 |
| DrogOff Offset V2 (17) | Similar | 23.54 | 26.51 | 46.61 | 38.58 | 88.53 | 97.87 | 21.14 | 100.00 | 37.87 |
| DrogOff Offset V2 (17) | Novel | 17.57 | 21.58 | 29.22 | 36.21 | 89.05 | 100.00 | 12.62 | 100.00 | 18.90 |
| DrogOff Native V3 LoRA (20) | Seen | 65.26 | 65.77 | 82.69 | 45.14 | 98.80 | 100.00 | 43.05 | 100.00 | 74.31 |
| DrogOff Native V3 LoRA (20) | Similar | 27.00 | 31.70 | 47.96 | 38.54 | 92.50 | 98.30 | 20.60 | 100.00 | 36.17 |
| DrogOff Native V3 LoRA (20) | Novel | 6.20 | 12.52 | 24.13 | 34.05 | 83.11 | 98.17 | 13.53 | 100.00 | 20.73 |
| CROG (21) | Seen | 61.81 | 61.80 | 79.79 | 44.97 | 98.26 | 98.42 | 41.43 | 100.00 | 71.94 |
| CROG (21) | Similar | 22.40 | 22.88 | 49.51 | 39.02 | 88.21 | 87.66 | 16.54 | 100.00 | 28.94 |
| CROG (21) | Novel | 8.00 | 11.96 | 22.07 | 33.43 | 79.80 | 81.71 | 9.09 | 100.00 | 9.76 |
| MapleGrasp (22) | Seen | 60.98 | 61.34 | 79.76 | 44.99 | 98.65 | 99.21 | 37.18 | 100.00 | 65.61 |
| MapleGrasp (22) | Similar | 11.41 | 14.70 | 36.11 | 36.13 | 84.31 | 87.23 | 14.11 | 99.57 | 20.94 |
| MapleGrasp (22) | Novel | 6.69 | 10.12 | 21.82 | 32.25 | 76.52 | 61.59 | 8.84 | 100.00 | 14.02 |
| ETRG (14) | Seen | 64.88 | 64.32 | 84.42 | 44.97 | 98.13 | 99.60 | 35.72 | 100.00 | 61.26 |
| ETRG (14) | Similar | 14.11 | 15.79 | 38.47 | 36.11 | 82.01 | 53.19 | 10.56 | 100.00 | 10.21 |
| ETRG (14) | Novel | 12.68 | 16.04 | 31.78 | 33.37 | 78.57 | 27.44 | 6.68 | 100.00 | 7.93 |
| LGD (23) | Seen | 43.88 | 46.36 | 64.47 | 42.49 | 94.70 | 81.03 | 20.29 | 100.00 | 37.55 |
| LGD (23) | Similar | -10.27 | -2.10 | 19.91 | 34.35 | 85.17 | 45.53 | 3.58 | 100.00 | 2.98 |
| LGD (23) | Novel | -8.11 | 0.50 | 17.20 | 33.35 | 86.37 | 44.51 | 2.56 | 100.00 | 4.27 |
| DrogOff Native V4 (36) | Seen | 52.53 | 52.54 | 74.71 | 44.14 | 96.84 | 94.47 | 27.61 | 100.00 | 31.23 |
| DrogOff Native V4 (36) | Similar | 13.08 | 15.91 | 42.03 | 36.16 | 85.71 | 48.51 | 11.34 | 100.00 | 6.81 |
| DrogOff Native V4 (36) | Novel | 14.60 | 16.81 | 33.08 | 35.13 | 82.55 | 29.88 | 3.92 | 100.00 | 3.05 |
| DrogOff Native V3 (19) | Seen | 53.09 | 53.43 | 74.41 | 43.63 | 96.84 | 97.23 | 27.79 | 100.00 | 27.27 |
| DrogOff Native V3 (19) | Similar | 18.98 | 19.29 | 45.15 | 38.45 | 89.31 | 91.49 | 14.42 | 100.00 | 11.06 |
| DrogOff Native V3 (19) | Novel | 14.95 | 18.22 | 30.99 | 35.32 | 83.92 | 64.63 | 6.41 | 100.00 | 3.05 |
| GRConvNet-CLIP (11) | Seen | 2.39 | 0.51 | 2.45 | 1.59 | 30.44 | 100.00 | 3.38 | 100.00 | 0.40 |
| GRConvNet-CLIP (11) | Similar | 2.30 | 0.28 | 2.53 | 1.64 | 32.51 | 100.00 | 0.35 | 100.00 | 0.00 |
| GRConvNet-CLIP (11) | Novel | 2.69 | 0.58 | 2.58 | 1.68 | 32.32 | 100.00 | 1.00 | 100.00 | 0.00 |
| GGCNN-CLIP (12) | Seen | 2.39 | 0.51 | 2.45 | 1.59 | 30.44 | 100.00 | 0.06 | 100.00 | 0.00 |
| GGCNN-CLIP (12) | Similar | 2.30 | 0.28 | 2.53 | 1.64 | 32.51 | 100.00 | 0.02 | 100.00 | 0.00 |
| GGCNN-CLIP (12) | Novel | 2.69 | 0.58 | 2.58 | 1.68 | 32.32 | 100.00 | 0.05 | 100.00 | 0.00 |

DrogOff Offset V2 leads Seen and Similar grasp accuracy. DrogOff Native V3
LoRA leads Novel grasp accuracy. GGCNN-CLIP and GRConvNet-CLIP are grasp-only
baselines whose wrappers reuse their grasp-quality map as a mask, so their
segmentation columns do not represent a genuine referring-segmentation head.

No complete independent three-split result is currently available for DROG,
CROG-OFF, standard DrogOff, or DrogOff Offset-Transport. Offset-Transport has
a training-end Seen artifact, but it is excluded until the selected checkpoint
is independently evaluated on Seen, Similar, and Novel.
