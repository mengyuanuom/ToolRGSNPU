# OCID-VLG, VCoT-GraspSet, and GraspNet-VLG test results

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

Grasp evaluation emits one rectangle with a fixed 40-pixel gripper depth.
`gAcc` requires best rotated IoU greater than 0.25 and angle error below
30 degrees; `Grasp mIoU` is the mean best grasp overlap over valid predictions.

All values below are percentages. Every completed row uses the experiment's
`best_iou` checkpoint and an independent evaluation on all three official
splits. The three split groups each contain only `Grasp mIoU` and **gAcc**;
grasp accuracy is bolded as the primary grasp-success metric.

### Selected comparison

<table>
  <thead>
    <tr>
      <th rowspan="2">Model (best-IoU epoch)</th>
      <th colspan="2">Seen</th>
      <th colspan="2">Similar</th>
      <th colspan="2">Novel</th>
    </tr>
    <tr>
      <th>Grasp mIoU</th>
      <th><strong>gAcc</strong></th>
      <th>Grasp mIoU</th>
      <th><strong>gAcc</strong></th>
      <th>Grasp mIoU</th>
      <th><strong>gAcc</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr><td>DrogOff Offset V2 (17)</td><td>44.62</td><td><strong>75.49</strong></td><td>21.14</td><td><strong>37.87</strong></td><td>12.62</td><td><strong>18.90</strong></td></tr>
    <tr><td>DrogOff Native V3 LoRA (20)</td><td>43.05</td><td><strong>74.31</strong></td><td>20.60</td><td><strong>36.17</strong></td><td>13.53</td><td><strong>20.73</strong></td></tr>
    <tr><td>CROG (21)</td><td>41.43</td><td><strong>71.94</strong></td><td>16.54</td><td><strong>28.94</strong></td><td>9.09</td><td><strong>9.76</strong></td></tr>
    <tr><td>MapleGrasp (22)</td><td>37.18</td><td><strong>65.61</strong></td><td>14.11</td><td><strong>20.94</strong></td><td>8.84</td><td><strong>14.02</strong></td></tr>
    <tr><td>ETRG (14)</td><td>35.72</td><td><strong>61.26</strong></td><td>10.56</td><td><strong>10.21</strong></td><td>6.68</td><td><strong>7.93</strong></td></tr>
    <tr><td>LGD (23)</td><td>20.29</td><td><strong>37.55</strong></td><td>3.58</td><td><strong>2.98</strong></td><td>2.56</td><td><strong>4.27</strong></td></tr>
    <tr><td>DrogOff Native V4 (36)</td><td>27.61</td><td><strong>31.23</strong></td><td>11.34</td><td><strong>6.81</strong></td><td>3.92</td><td><strong>3.05</strong></td></tr>
    <tr><td>DrogOff Native V3 (19)</td><td>27.79</td><td><strong>27.27</strong></td><td>14.42</td><td><strong>11.06</strong></td><td>6.41</td><td><strong>3.05</strong></td></tr>
    <tr><td>GRConvNet-CLIP (11)</td><td>3.38</td><td><strong>0.40</strong></td><td>0.35</td><td><strong>0.00</strong></td><td>1.00</td><td><strong>0.00</strong></td></tr>
    <tr><td>GGCNN-CLIP (12)</td><td>0.06</td><td><strong>0.00</strong></td><td>0.02</td><td><strong>0.00</strong></td><td>0.05</td><td><strong>0.00</strong></td></tr>
  </tbody>
</table>
DrogOff Offset V2 leads Seen and Similar grasp accuracy. DrogOff Native V3
LoRA leads Novel grasp accuracy.

No complete independent three-split result is currently available for DROG,
CROG-OFF, standard DrogOff, or DrogOff Offset-Transport. Offset-Transport has
a training-end Seen artifact, but it is excluded until the selected checkpoint
is independently evaluated on Seen, Similar, and Novel.
