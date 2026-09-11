# ETRG / GGCNN corrected GraspTools V3 training

Use these new configs with the existing NPU launcher:

- `config/grasp_tools/etrg_v3_sigmoid_masked.yaml`
- `config/grasp_tools/ggcnnclip_v3_sigmoid_masked.yaml`

Both enable `TRAIN.grasp_loss_profile: sigmoid_masked`. Quality and width
use sigmoid for loss computation and decoding. Quality uses balanced
positive/negative Smooth L1; sine, cosine, width and optional short-side
regression use only pixels with labelled width above 1e-6. Angles remain
unbounded doubled-angle sine/cosine regression, not sigmoid.

Configs without an explicit loss profile default to `legacy`.
The standard VCOT, OCID-VLG (including ETRG R50/R101 composed configs), and
RealVLG/GraspNet-VLG ETRG/GGCNN configs now explicitly enable `sigmoid_masked`.
VCOT ETRG also has a convenient `config/vcot/etrg.yaml` entry point.
Their dataset splits, size factors and evaluation protocols are unchanged;
their default experiment names now end in `_sigmoid_masked`.
For old checkpoints use the original config from before this update, not
these updated training configs. Existing running processes are not changed.
Do not evaluate old weights with the new profiles. Resume across loss profiles
is rejected; start a fresh experiment for the correction. Checkpoints record
the loss profile and decoding activation. For evaluation pass the matching
training config, not merely an architecture name. Changing only decoding
cannot repair weights trained under the old objectives.

New profiles use V3 original-coordinate factor-300 labels and primary
IoU 0.25 / angle 30 degrees. Additional IoU thresholds are 0.50 and 0.75.
This is NOT the complete 12-pair GPU mSR evaluator: the existing NPU loop
reports multiple IoUs at a single angle threshold.

Global batch sizes inherit NPU launcher semantics. Tune per available device
count; these settings do not claim identical GPU/NPU optimization budgets.
No new training has been started by this code update.

CPU numerical tests: `python -m unittest discover -s tests -p test_sigmoid_grasp_loss.py`.
Run an NPU forward/backward smoke test before a full training run.
