"""Convert dense grasp maps into explicit rotated grasp candidates."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from skimage.feature import peak_local_max

from toolrgs.registry import POSTPROCESSORS


@dataclass(frozen=True)
class GraspDetection:
    x: float
    y: float
    width: float
    height: float
    angle_degrees: float
    score: float
    row: int
    column: int

    def as_rectangle(self):
        return [self.x, self.y, self.width, self.height, self.angle_degrees]


@POSTPROCESSORS.register_module(name="dense_grasp", aliases=("peak_grasp",))
class DenseGraspPostProcessor:
    def __init__(
        self,
        quality_threshold: float = 0.4,
        min_distance: int = 2,
        num_grasps: int = 1,
        width_factor: float = 100.0,
        grasp_height: float = 20.0,
    ):
        self.quality_threshold = float(quality_threshold)
        self.min_distance = int(min_distance)
        self.num_grasps = int(num_grasps)
        self.width_factor = float(width_factor)
        self.grasp_height = float(grasp_height)

    def __call__(
        self,
        quality,
        sine,
        cosine,
        width,
        num_grasps: Optional[int] = None,
        spatial_scale: float = 1.0,
    ):
        quality = np.asarray(quality, dtype=np.float32)
        sine = np.asarray(sine, dtype=np.float32)
        cosine = np.asarray(cosine, dtype=np.float32)
        width = np.asarray(width, dtype=np.float32)
        if not (quality.shape == sine.shape == cosine.shape == width.shape):
            raise ValueError("quality/sine/cosine/width maps must share one shape")
        if quality.ndim != 2:
            raise ValueError(f"Dense grasp maps must be 2-D, got {quality.shape}")
        count = self.num_grasps if num_grasps is None else int(num_grasps)
        peaks = peak_local_max(
            quality,
            min_distance=self.min_distance,
            threshold_abs=self.quality_threshold,
            num_peaks=count,
        )
        angle = np.arctan2(sine, cosine) / 2.0
        scale = float(spatial_scale)
        return [
            GraspDetection(
                x=float(column),
                y=float(row),
                width=max(1.0, float(width[row, column]) * self.width_factor * scale),
                height=self.grasp_height * scale,
                angle_degrees=float(angle[row, column] / np.pi * 180.0),
                score=float(quality[row, column]),
                row=int(row),
                column=int(column),
            )
            for row, column in peaks
        ]
