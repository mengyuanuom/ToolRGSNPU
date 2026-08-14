# VCoT-Grasp paper-compatible protocol

Every profile under `config/vcot/` uses the same VCoT-Grasp comparison
contract:

- `train.csv` excluding its final 5,000 rows is used for training;
- the final 5,000 `train.csv` rows are used for checkpoint validation;
- only the first (highest-ranked) grasp label supervises each training sample;
- grasp length and width are normalized in original 416-pixel coordinates;
- evaluation uses one prediction, all ground-truth grasps, continuous OpenCV
  rotated IoU, IoU `>= 0.25`, and 180-degree-periodic angle error `<= 30`;
- predicted grasps are not filtered by the segmentation mask.

The public VCoT benchmark reports seen and unseen test sets separately. Run
both with the same checkpoint:

```bash
python evaluate.py \
  --config config/vcot/drogoff.yaml \
  --checkpoint exp/vcot/drogoff_vcot/best_model.pth \
  --split seen \
  --npu 0

python evaluate.py \
  --config config/vcot/drogoff.yaml \
  --checkpoint exp/vcot/drogoff_vcot/best_model.pth \
  --split unseen \
  --npu 0
```

Use `GraspSR(IoU=0.25)` for comparison with published VCoT-Grasp results.
Segmentation IoU and precision are ToolRGS auxiliary metrics and should be
reported separately. To add a stricter supplementary result without changing
the paper-comparison metric, override `TEST.grasp_iou_thresholds` and keep
`TEST.grasp_iou_threshold: 0.25` as the primary checkpoint-selection metric.
