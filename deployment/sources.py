"""Camera sources used by the deployment GUI."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from toolrgs.registry import CAMERAS


class FrameSource:
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class ImageSource(FrameSource):
    def __init__(self, image_path: str):
        self.path = Path(image_path)
        self.image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Could not read image: {self.path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return True, self.image.copy()


class OpenCVSource(FrameSource):
    def __init__(
        self,
        source: Any,
        width: int = 0,
        height: int = 0,
        fps: int = 0,
        api_preference: int = cv2.CAP_ANY,
    ):
        self.capture = cv2.VideoCapture(source, api_preference)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open OpenCV source: {source}")
        if width:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps:
            self.capture.set(cv2.CAP_PROP_FPS, int(fps))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.capture.read()

    def close(self) -> None:
        self.capture.release()


class RealSenseSource(FrameSource):
    def __init__(self, width: int, height: int, fps: int):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense backend requires pyrealsense2; install requirement-deploy.txt"
            ) from exc
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        self.pipeline.start(config)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data()).copy()

    def close(self) -> None:
        self.pipeline.stop()


class GStreamerSource(FrameSource):
    """GI/GStreamer appsink source matching the shared-memory server pipeline."""

    def __init__(self, pipeline_text: str):
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except (ImportError, ValueError) as exc:
            raise RuntimeError("PyGObject/GStreamer is unavailable") from exc
        self.Gst = Gst
        Gst.init(None)
        if "name=toolrgs_sink" not in pipeline_text:
            index = pipeline_text.rfind("appsink")
            if index < 0:
                raise ValueError("GStreamer pipeline must end in an appsink")
            index += len("appsink")
            pipeline_text = pipeline_text[:index] + " name=toolrgs_sink" + pipeline_text[index:]
        self.pipeline = Gst.parse_launch(pipeline_text)
        self.sink = self.pipeline.get_by_name("toolrgs_sink")
        if self.sink is None:
            raise RuntimeError("Could not find GStreamer appsink named toolrgs_sink")
        self.sink.set_property("sync", False)
        self.sink.set_property("drop", True)
        self.sink.set_property("max-buffers", 1)
        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer pipeline failed to enter PLAYING state")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        sample = self.sink.emit("try-pull-sample", 3 * self.Gst.SECOND)
        if sample is None:
            return False, None
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        pixel_format = str(caps.get_value("format"))
        channels = {"BGR": 3, "RGB": 3, "BGRA": 4, "RGBA": 4}.get(pixel_format)
        if channels is None:
            raise RuntimeError(f"Unsupported GStreamer pixel format: {pixel_format}")
        buffer = sample.get_buffer()
        success, info = buffer.map(self.Gst.MapFlags.READ)
        if not success:
            return False, None
        try:
            raw = np.frombuffer(info.data, dtype=np.uint8)
            row_stride = raw.size // int(height)
            packed = raw.reshape(int(height), row_stride)[:, : int(width) * channels]
            frame = packed.reshape(int(height), int(width), channels).copy()
        finally:
            buffer.unmap(info)
        if pixel_format == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif pixel_format == "RGBA":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif pixel_format == "BGRA":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return True, frame

    def close(self) -> None:
        self.pipeline.set_state(self.Gst.State.NULL)


@CAMERAS.register_module(name="image")
def _build_image_source(camera_cfg: Dict[str, Any], repo_root: str) -> FrameSource:
    value = camera_cfg.get("image_path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(repo_root) / path
    return ImageSource(str(path))


@CAMERAS.register_module(name="realsense", aliases=("intel_realsense",))
def _build_realsense_source(camera_cfg: Dict[str, Any], repo_root: str) -> FrameSource:
    del repo_root
    return RealSenseSource(
        int(camera_cfg.get("width", 1280)),
        int(camera_cfg.get("height", 720)),
        int(camera_cfg.get("fps", 30)),
    )


@CAMERAS.register_module(name="gstreamer", aliases=("gst",))
def _build_gstreamer_source(camera_cfg: Dict[str, Any], repo_root: str) -> FrameSource:
    del repo_root
    pipeline = str(camera_cfg.get("gstreamer_pipeline", "")).strip()
    if not pipeline:
        raise ValueError("camera.gstreamer_pipeline is required for gstreamer")
    try:
        return GStreamerSource(pipeline)
    except RuntimeError as gi_error:
        try:
            return OpenCVSource(pipeline, api_preference=cv2.CAP_GSTREAMER)
        except RuntimeError as cv_error:
            raise RuntimeError(
                f"GStreamer failed through PyGObject ({gi_error}) and OpenCV ({cv_error})"
            ) from cv_error


@CAMERAS.register_module(name="video")
def _build_video_source(camera_cfg: Dict[str, Any], repo_root: str) -> FrameSource:
    value = Path(str(camera_cfg.get("video_path", ""))).expanduser()
    if not value.is_absolute():
        value = Path(repo_root) / value
    return OpenCVSource(str(value))


@CAMERAS.register_module(name="opencv", aliases=("camera", "usb"))
def _build_opencv_source(camera_cfg: Dict[str, Any], repo_root: str) -> FrameSource:
    del repo_root
    value = camera_cfg.get("device", 0)
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    width = int(camera_cfg.get("width", 0))
    height = int(camera_cfg.get("height", 0))
    fps = int(camera_cfg.get("fps", 0))
    return OpenCVSource(value, width, height, fps)


CAMERA_REGISTRY = CAMERAS.module_dict


def build_source(camera_cfg: Dict[str, Any], repo_root: str) -> FrameSource:
    """Build a registered camera from either ``type`` or legacy ``backend``."""
    component_type = camera_cfg.get("type", camera_cfg.get("backend", "opencv"))
    try:
        factory = CAMERAS.require(component_type)
    except KeyError as exc:
        available = ", ".join(sorted(CAMERAS.keys()))
        raise ValueError(
            f"Unknown camera component {component_type!r}; available: {available}"
        ) from exc
    return factory(camera_cfg, repo_root)
