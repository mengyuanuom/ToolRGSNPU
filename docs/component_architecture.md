# ToolRGSNPU component architecture

ToolRGS is migrating incrementally to an MMDetection-style architecture. The
first stage introduces shared registries and named model results without
breaking existing experiment YAML, checkpoints, imports, or training commands.

## Registries

The global registries live in `toolrgs/registry.py`:

```text
MODELS          DATASETS        TRANSFORMS
LOSSES          METRICS         POSTPROCESSORS
LOOPS           HOOKS           CAMERAS
ROBOT_CLIENTS   DETECTORS       AUDIO_INPUTS
```

Names are case-insensitive and normalize spaces/hyphens to underscores. Both
decorator and direct registration are supported:

```python
from toolrgs.registry import CAMERAS

@CAMERAS.register_module(name="my_camera", aliases=("lab-camera",))
def build_my_camera(camera_cfg, repo_root):
    return MyCamera(camera_cfg["device"])
```

The deployment config can then select it without changing a central builder:

```yaml
camera:
  type: my_camera
  device: 0
```

Run the component inventory in a configured training environment:

```bash
python tools/list_components.py
python tools/list_components.py --group models
```

## Model output contract

New code should use the structures in `toolrgs/structures`:

```python
GraspOutput(
    segmentation=seg,
    quality=quality,
    sine=sine,
    cosine=cosine,
    width=width,
    offset=offset,  # optional
)
```

`GraspModelResult` groups predictions, targets, the scalar training loss, and
named loss terms. `GraspModelResult.from_legacy(...)` accepts all historical
ToolRGS return layouts:

```text
(seg, quality, sine, cosine, width)
(seg, quality, sine, cosine, width, offset)
(predictions, targets)
(predictions, targets, total_loss, loss_dict)
```

Deployment inference already normalizes legacy model results through this
structure. `LegacyOutputAdapter` lets new loops consume an existing model as a
structured-output module, while the current training engine continues receiving
its historical tuples.

## Loops, hooks, metrics, and postprocessing

The main trainer now runs training and validation epochs through the registered
`GraspTrainLoop` and `GraspValLoop`.
The loop owns device transfer, AMP, backward/optimizer steps, distributed meter
reduction, and progress logging. The CLI still owns experiment construction,
checkpointing, validation scheduling, and scheduler stepping. The optional flat
config keys `RUNTIME.train_loop` and `RUNTIME.val_loop` select other registered
loops; they default to `grasp_train` and `grasp_val` for all existing YAML
files. Validation hooks can be configured independently with
`RUNTIME.val_hooks`.

Hooks receive a mutable `LoopState` at `before_epoch`, `before_iter`,
`after_iter`, and `after_epoch`. They are ordered by numeric priority:

```yaml
RUNTIME:
  hooks:
    - type: noop
```

Registered evaluation components currently include:

- `BinarySegmentationMetric`: mean per-sample IoU and precision thresholds;
- `GraspSuccessMetric`: top-k Jacquard success aggregation;
- `DenseGraspPostProcessor`: quality-peak decoding into named rotated grasps.

Validation and real-world deployment both use `DenseGraspPostProcessor`.
`GraspValLoop` also owns inverse affine warping, optional offset refinement,
per-sample segmentation metrics, top-1/top-5 Jacquard evaluation, and
distributed reduction of sufficient statistics.

## Compatibility layer

- `model.MODEL_REGISTRY` remains available as a read-only view of `MODELS`.
- `utils.data_builder.DATASET_REGISTRY` remains available as a read-only view of
  `DATASETS`.
- `build_model(cfg)` still returns `(model, optimizer_parameter_groups)`.
- `build_dataset(cfg, split, with_offset)` keeps its existing signature.
- `engine.engine.validate_with_grasp(...)` remains as a compatibility wrapper
  around `GraspValLoop`; the previous implementation is retained privately as
  `_legacy_validate_with_grasp` for short-term parity diagnosis.
- Dataset-specific optional arguments are signature-filtered; custom registered
  datasets can receive additional values through `DATA.dataset_args`.
- Deployment YAML accepts the new `type` field and still understands the old
  camera `backend` field.

## Migration sequence

The safe next stages are:

1. extract optimizer behavior into an `OptimWrapper` and checkpoint/logging into
   concrete hooks;
2. register transforms and losses;
3. convert each model to `BaseGraspModel` and remove tuple adapters;
4. introduce composable `_base_` experiment configs after all legacy configs
   have parity tests.

At every stage the old CLI remains a compatibility entry until equivalent
training/evaluation results have been checked.
