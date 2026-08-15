# VCoT alignment and evaluation protocol

This port uses two explicit references:

- ToolRGS `422dd4fc066a209c09bb884ca6c345e9b9dcd308` for dense-model architecture, grasp-size training, and MapleGrasp staging.
- Official VCoT-Grasp `8fea4b3887f3d582970dd68ce0cb252b5bb7e2d8` for dataset splits and grasp-success evaluation.

## Training alignment

The VCoT dense baselines use their model-specific canvas size, a grasp-size factor of 300, canvas-coordinate width/short-side targets, and all positive grasp labels. Predicted sizes are restored once from canvas coordinates to source-image coordinates during decoding.

Every grasp-producing VCoT model regresses raw long- and short-side logits, applies sigmoid to both logits in the training loss, and applies sigmoid once during evaluation/deployment. The auto size activation resolves to sigmoid from model/checkpoint metadata.

CROG, CROG-OFF, DROG, DROG-OFF, GGCNN-CLIP, GR-ConvNet-CLIP, GraspMamba, LGD, MapleGrasp, and ETRG own their short-side branch inside the native decoder/projector. The model builder no longer attaches the generic fallback adapter. MapleGrasp Stage 1 remains segmentation-only; Stage 2 owns the grasp and short-side heads.

This training target is intentionally different from the official VCoT-Grasp foundation model, whose action head trains on the highest-score grasp only. ToolRGS dense baselines train dense maps from all positive labels. Evaluation is shared and is described below.

## Official evaluation

`evaluation_protocol: vcot_official` enforces:

- the official `test_seen.csv` and `test_unseen.csv` files;
- exactly one predicted grasp per sample;
- comparison against every ground-truth grasp for that sample;
- continuous OpenCV rotated-rectangle IoU greater than or equal to 0.25;
- 180-degree-periodic angle error less than or equal to 30 degrees;
- failed or missing predictions remaining in the denominator;
- no segmentation-mask filtering of the predicted grasp center.

The repository copies of the official test files are byte-identical to VCoT-Grasp: 3,000 seen samples and 1,487 unseen samples. During training, the official final 5,000 rows of `train.csv` are held out as validation data by `train_official` and `val_official`.

Evaluate a checkpoint on both test subsets:

```bash
python evaluate.py --config config/vcot/drogoff.yaml --checkpoint <checkpoint.pth> --npu 0 --split seen
python evaluate.py --config config/vcot/drogoff.yaml --checkpoint <checkpoint.pth> --npu 0 --split unseen
```

The reported grasp metric is `GraspSR`, not ToolRGS top-k `J@1/J@5`.

## Checkpoint compatibility

Checkpoints trained with an earlier NPU VCoT protocol are not directly comparable with this alignment. Earlier profiles used different size factors/coordinates or target selection, and several models used a generic short-side adapter or clamp decoding. Retrain the aligned profiles; native-head tensor shapes and saved sigmoid metadata intentionally prevent silently evaluating an incompatible checkpoint.
