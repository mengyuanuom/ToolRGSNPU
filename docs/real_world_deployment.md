# Real-world ToolRGSNPU deployment

This deployment layer replaces the server CROG demo's hard-coded model, paths,
camera pipeline, IP address, and automatic TCP sends with one YAML file. It
supports every ToolRGS grasp architecture, OpenCV/video, Intel
RealSense, shared-memory GStreamer, an optional MMDetection tab, and optional
Whisper microphone input.

Supported grasp models are CROG, CROGOFF, DROG, DROGOFF, ETRG-A, MapleGrasp,
GGCNN-CLIP, GR-ConvNet-CLIP, GraspMamba, and LGD. DETRIS is intentionally excluded here: in
this repository it is a segmentation/backbone component, not a grasp-map model.

## 1. Install

Use the same CANN/torch_npu environment as training, then install the GUI and
camera extras:

```bash
pip install -r requirement.txt
pip install -r requirement-deploy.txt
```

For GStreamer on Ubuntu, install the system GI bindings and plugins. PyGObject
is normally installed through `apt`, not `pip`:

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

Whisper also needs `ffmpeg`. Object detection is optional and needs an
MMCV/MMDetection build adapted for the installed Ascend, CANN, and PyTorch versions;
it is intentionally not included in the base requirements.

## 2. Put weights in place

Training, CLIP/DINO/Mamba, detector, and Whisper weights are not committed to
Git. Copy the files referenced by the selected experiment and deployment YAML.
For example:

```text
pretrain/ViT-B-16.pt
pretrain/dinov2_vitb14_reg4_pretrain.pth
exp/grasp_tools/drogoff_grasp_tools/best_jindex_model.pth
weights/epoch_48_13.pth                 # only when detector.enabled=true
```

Paths in deployment YAML are resolved from the ToolRGS repository root, so the
command can be run from any working directory.

## 3. Create a lab config and preflight it

```bash
cp config/deployment/lab.example.yaml config/deployment/lab.yaml
# Edit model/checkpoint, camera backend, camera path/device, and prompt.
python tools/check_deployment.py --config config/deployment/lab.yaml
python tools/check_deployment.py --config config/deployment/lab.yaml \
  --probe-camera --build-model
```

The preflight never connects to the robot and never sends a command.

Camera components are selected with `camera.type` (`camera.backend` remains a
legacy alias):

- `opencv`: integer USB camera index, RTSP URL, or other OpenCV source.
- `video`: repository-relative or absolute video path.
- `image`: repeat one image; useful for a safe end-to-end GUI check.
- `realsense`: direct color stream through `pyrealsense2`.
- `gstreamer`: shared-memory or network pipeline ending in `appsink`. Start from
  `config/deployment/gstreamer.example.yaml` for the old CROG `shmsrc` layout.

## 4. Dry-run the GUI

Keep `robot.enabled: false` and run:

```bash
python deploy_gui.py --config config/deployment/lab.yaml
```

Check the segmentation overlay, grasp rectangle, center, angle, and width before
using a physical robot. The object detector and audio controls only appear when
their respective `enabled` settings are true.

## 5. Receiver and coordinate contract

The server CROG snapshot does **not** contain the Kinova-side receiver, robot
motion controller, hand-eye calibration, workspace limits, or collision logic.
ToolRGS therefore supplies the compatible sender, not those missing components.
The external receiver must already be running and validated.

The sender emits one ASCII line per command:

```text
{x, y, theta, width, depth}\n
```

- `x`, `y`: grasp center in the configured coordinate space.
- `theta`: image-plane grasp angle in degrees.
- `width`: gripper rectangle width in pixels.
- `depth`: the old demo's semantic tier (`-1`, `0`, or `1`), **not a RealSense
  depth measurement**.

Set `robot.coordinate_space` to match the receiver calibration:

- `source`: original camera pixels after inverse letterbox; recommended for a
  new calibration.
- `model`: letterboxed model-input pixels; useful for a receiver calibrated in
  ToolRGS input space.

Do not guess this setting. A receiver calibrated against the historic demo's
stretched 416x416 image is not automatically equivalent to either mapping and
must be recalibrated or given an explicit compatibility transform.

`robot.limits` is a second sender-side guard for center, angle, width, and depth.
Update these bounds for the configured coordinate space; an out-of-range command
is rejected before any bytes are sent.

## 6. Enable physical output

After verifying the receiver, calibration, robot limits, emergency stop, and
dry-run prediction:

1. Set `robot.enabled: true`, the correct `host`/`port`, and leave
   `auto_send: false`.
2. Launch with the additional command-line permission:

   ```bash
   python deploy_gui.py --config config/deployment/lab.yaml --allow-robot
   ```

3. Click **Connect receiver**, then explicitly tick **Arm robot output**.
4. Use **Send current grasp** for the first tests.

Automatic sending additionally requires `auto_send: true` and an armed GUI;
`auto_send_interval_s` rate-limits it. Manual sending is recommended until the
whole calibrated workspace has been tested.

## Ported and intentionally excluded pieces

The reusable pieces from the local server snapshot are now modules: grasp GUI,
shared-memory/direct camera access, optional 13-class detector, Whisper input,
semantic depth tiers, and the legacy TCP sender. Absolute `/home/...` paths,
fixed accelerator devices, Qt plugin paths, global `torch.load` monkey patches, and
automatic every-50-frame sends were removed.

The experimental GelSight scripts were not copied into the default deployment:
their classifier weights and a stable sensor/label contract are absent from the
snapshot. They can be added as an optional adapter once those artifacts and the
intended GelSight task are provided, without changing the grasp/robot path.
