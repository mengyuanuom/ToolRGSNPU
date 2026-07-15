# ToolRGSNPU component architecture

ToolRGSNPU now uses the second-stage MMDetection-style architecture. Shared
registries and named results remain compatible with historical models, while a
Runner, OptimWrapper, parameter-scheduler registry, checkpoint hooks, and
composable `_base_` configs own new experiments.

## Registries

The global registries live in `toolrgs/registry.py`:

```text
MODELS          DATASETS        TRANSFORMS
LOSSES          METRICS         POSTPROCESSORS
RUNNERS         LOOPS           HOOKS
OPTIM_WRAPPERS  PARAM_SCHEDULERS
CAMERAS
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

`NPUGraspRunner` owns distributed initialization, component construction,
dataset loaders, train/validation sequencing, resume, and shutdown.
`GraspTrainLoop` and `GraspValLoop` remain independently registered. The
`NPUAmpOptimWrapper` owns scaled backward, gradient clipping, and stepping;
`CheckpointHook` and `LoggerHook` own epoch persistence and summaries.

```yaml
RUNTIME:
  runner: {type: npu_grasp}
  optim_wrapper: {type: npu_amp}
  param_scheduler:
    type: multi_step
    milestones: [35]
    gamma: 0.1
  runner_hooks:
    - {type: logger}
    - {type: checkpoint}
```

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

## Composable configuration

`utils.config` supports one or several relative `_base_` YAML files, recursive
deep merge, circular-inheritance detection, and MMEngine-style `_delete_` for
mapping replacement. It keeps both views:

- `cfg.sections.MODEL`, `cfg.sections.DATA`, etc. preserve hierarchy;
- `cfg.architecture`, `cfg.root_path`, etc. keep historical models runnable.

The preferred layout is:

```text
configs/
├── _base_/
│   ├── datasets/
│   ├── models/
│   ├── schedules/
│   └── runtime/
└── etrg/
    ├── etrg_r50_ocid_vlg.yaml
    └── etrg_r101_ocid_vlg.yaml
```

Run a composed experiment with:

```bash
python tools/train.py --config configs/etrg/etrg_r50_ocid_vlg.yaml
```

The root `python train.py --config ...` command and files under `config/`
remain supported.

The canonical Runner import is now `toolrgs.datasets.build_dataset`; it bridges
to the historical `utils.data_builder` implementation until individual dataset
classes move into `toolrgs/datasets/`.

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

## Remaining migration sequence

The safe next stages are:

1. move dataset preprocessing into registered transform pipelines;
2. convert each historical model to `BaseGraspModel` and remove tuple adapters;
3. migrate the remaining legacy experiment files from `config/` to `configs/`;
4. add a Runner-owned test loop and deployment data preprocessor.

At every stage the old CLI remains a compatibility entry until equivalent
training/evaluation results have been checked.
