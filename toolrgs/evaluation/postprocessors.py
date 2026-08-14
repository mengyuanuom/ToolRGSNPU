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
        minimum_width: float = 1.0,
        size_coordinate: str = "original",
    ):
        self.quality_threshold = float(quality_threshold)
        self.min_distance = int(min_distance)
        self.num_grasps = int(num_grasps)
        self.width_factor = float(width_factor)
        self.grasp_height = float(grasp_height)
        self.minimum_width = float(minimum_width)
        self.size_coordinate = str(size_coordinate).strip().lower()
        if self.size_coordinate not in {"original", "canvas"}:
            raise ValueError(
                "size_coordinate must be 'original' or 'canvas', got "
                f"{size_coordinate!r}"
            )

    def __call__(
        self,
        quality,
        sine,
        cosine,
        width,
        num_grasps: Optional[int] = None,
        spatial_scale: float = 1.0,
        short_side=None,
    ):
        quality = np.asarray(quality, dtype=np.float32)
        sine = np.asarray(sine, dtype=np.float32)
        cosine = np.asarray(cosine, dtype=np.float32)
        width = np.asarray(width, dtype=np.float32)
        short_side = (
            None if short_side is None else np.asarray(short_side, dtype=np.float32)
        )
        geometry_maps = [quality, sine, cosine, width]
        if short_side is not None:
            geometry_maps.append(short_side)
        if any(value.shape != quality.shape for value in geometry_maps[1:]):
            raise ValueError("dense grasp geometry maps must share one shape")
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
        scale = (
            float(spatial_scale)
            if self.size_coordinate == "canvas"
            else 1.0
        )
        return [
            GraspDetection(
                x=float(column),
                y=float(row),
                width=max(
                    self.minimum_width,
                    float(width[row, column]) * self.width_factor * scale,
                ),
                height=(
                    max(
                        self.minimum_width,
                        float(short_side[row, column]) * self.width_factor * scale,
                    )
                    if short_side is not None
                    else self.grasp_height
                ),
                angle_degrees=float(angle[row, column] / np.pi * 180.0),
                score=float(quality[row, column]),
                row=int(row),
                column=int(column),
            )
            for row, column in peaks
        ]
