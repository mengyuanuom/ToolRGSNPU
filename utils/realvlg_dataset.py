"""RealVLG-11B GraspNet adapter for ToolRGS.

The public RealVLG-R1 evaluator defines the benchmark on ``GraspNet_VLG``:

* metadata lives under ``metadata/<camera>/<scene>/<frame>.json``;
* evaluation uses frame ``0000.json`` only;
* scene ranges are 0100-0129 (seen), 0130-0159 (similar), and
  0160-0189 (novel).

ToolRGS requires a fixed-size tensor, so images are deterministically
letterboxed while preserving aspect ratio. Predictions are evaluated back in
the untouched source-image coordinate system.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional, Set

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.dataset import GraspTransforms, make_grasp_offset_targets_np, tokenize


REALVLG_EVAL_SCENES = {
    "seen": range(100, 130),
    "similar": range(130, 160),
    "novel": range(160, 190),
}

_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "seen": "seen",
    "similar": "similar",
    "novel": "novel",
}


def resolve_realvlg_split(split: str) -> str:
    key = str(split).strip().lower().replace("-", "_")
    try:
        return _SPLIT_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_SPLIT_ALIASES))
        raise ValueError(
            f"Unknown RealVLG split {split!r}; choose one of: {choices}"
        ) from exc


def realvlg_points8_to_rectangles(points8: np.ndarray) -> np.ndarray:
    """Convert official ``[x0,y0,...,x3,y3]`` grasps to ToolRGS rectangles.

    RealVLG defines the gripper opening along edge 0->1. This differs from the
    corner ordering assumed by some historical ToolRGS datasets, so the
    conversion is intentionally implemented here instead of reusing their
    rectangle parser.
    """

    values = np.asarray(points8, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim == 3 and values.shape[1:] == (4, 2):
        corners = values
    elif values.ndim == 2 and values.shape[1] == 8:
        corners = values.reshape(-1, 4, 2)
    else:
        raise ValueError(
            "RealVLG grasps must have shape [N,8] or [N,4,2], "
            f"got {values.shape}"
        )

    center = corners.mean(axis=1)
    left_midpoint = corners[:, [0, 3]].mean(axis=1)
    right_midpoint = corners[:, [1, 2]].mean(axis=1)
    top_midpoint = corners[:, [0, 1]].mean(axis=1)
    bottom_midpoint = corners[:, [2, 3]].mean(axis=1)
    opening_width = np.linalg.norm(right_midpoint - left_midpoint, axis=1)
    gripper_depth = np.linalg.norm(bottom_midpoint - top_midpoint, axis=1)
    edge = corners[:, 1] - corners[:, 0]
    angle_degrees = np.rad2deg(np.arctan2(edge[:, 1], edge[:, 0]))
    target = np.zeros(len(corners), dtype=np.float32)
    return np.stack(
        [
            center[:, 0],
            center[:, 1],
            opening_width,
            gripper_depth,
            angle_degrees,
            target,
        ],
        axis=1,
    ).astype(np.float32)


def _stable_rank(sample_key: str, seed: int) -> int:
    payload = f"{int(seed)}:{sample_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class RealVLGDataset(Dataset):
    """Load RealVLG GraspNet images, descriptions, masks, and 2-D grasps."""

    def __init__(
        self,
        root_dir,
        input_size=448,
        split="train",
        word_length=77,
        camera_mode="kinect",
        with_depth=False,
        with_offset=False,
        offset_radius=20.0,
        offset_sigma=None,
        offset_version="v1",
        offset_target_stride=4,
        offset_weight_floor=0.25,
        train_fraction=0.1,
        train_seed=0,
        train_manifest=None,
        eval_frame="0000",
        width_factor=100.0,
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.input_size = (int(input_size), int(input_size))
        self.word_length = int(word_length)
        self.camera_mode = str(camera_mode)
        self.split = resolve_realvlg_split(split)
        self.with_depth = bool(with_depth)
        self.with_offset = bool(with_offset)
        self.offset_radius = float(offset_radius)
        self.offset_sigma = offset_sigma
        self.offset_version = str(offset_version).strip().lower()
        if self.offset_version not in {"v1", "v2"}:
            raise ValueError(
                f"offset_version must be 'v1' or 'v2', got {offset_version!r}"
            )
        self.offset_target_stride = int(offset_target_stride)
        if self.offset_target_stride <= 0:
            raise ValueError("offset_target_stride must be positive")
        self.offset_weight_floor = float(offset_weight_floor)
        if not 0.0 <= self.offset_weight_floor <= 1.0:
            raise ValueError("offset_weight_floor must be between 0 and 1")
        self.train_fraction = float(train_fraction)
        self.train_seed = int(train_seed)
        eval_frame_value = str(eval_frame)
        self.eval_frame = (
            eval_frame_value[:-5]
            if eval_frame_value.endswith(".json")
            else eval_frame_value
        )
        self.width_factor = float(width_factor)
        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], dtype=torch.float32
        ).reshape(3, 1, 1)
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], dtype=torch.float32
        ).reshape(3, 1, 1)
        self.grasp_transform = GraspTransforms(
            width_factor=self.width_factor,
            width=self.input_size[1],
            height=self.input_size[0],
        )

        if self.with_depth:
            raise ValueError(
                "The public RealVLG GraspNet metadata contract does not expose an "
                "aligned depth_path. Set DATA.with_depth=false for this adapter."
            )
        if not 0.0 < self.train_fraction <= 1.0:
            raise ValueError(
                f"train_fraction must be in (0,1], got {self.train_fraction}"
            )
        self.metadata_dir = self.root_dir / "metadata" / self.camera_mode
        if not self.metadata_dir.is_dir():
            raise FileNotFoundError(
                f"RealVLG metadata directory not found: {self.metadata_dir}"
            )

        manifest_keys = self._load_manifest(train_manifest)
        self.samples = self._load_samples(manifest_keys)
        if not self.samples:
            raise ValueError(
                f"No usable RealVLG samples found for split={self.split!r} "
                f"under {self.metadata_dir}"
            )

    @staticmethod
    def _load_manifest(path) -> Optional[Set[str]]:
        if not path:
            return None
        manifest_path = Path(path).expanduser()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"RealVLG training manifest not found: {manifest_path}"
            )
        keys = {
            line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not keys:
            raise ValueError(f"RealVLG training manifest is empty: {manifest_path}")
        return keys

    def _metadata_files(self) -> Iterable[Path]:
        if self.split == "train":
            for scene_id in range(0, 100):
                scene_dir = self.metadata_dir / f"scene_{scene_id:04d}"
                if scene_dir.is_dir():
                    yield from sorted(scene_dir.glob("*.json"))
            return
        for scene_id in REALVLG_EVAL_SCENES[self.split]:
            path = (
                self.metadata_dir
                / f"scene_{scene_id:04d}"
                / f"{self.eval_frame}.json"
            )
            if path.is_file():
                yield path

    def _load_samples(self, manifest_keys: Optional[Set[str]]):
        samples = []
        for metadata_path in self._metadata_files():
            relative_metadata = metadata_path.relative_to(self.metadata_dir).as_posix()
            with metadata_path.open("r", encoding="utf-8") as stream:
                objects = json.load(stream)
            if not isinstance(objects, list):
                raise ValueError(
                    f"RealVLG metadata must contain a list: {metadata_path}"
                )
            for object_index, item in enumerate(objects):
                if not isinstance(item, dict) or not item.get("grasps"):
                    continue
                object_id = str(item.get("object_id", object_index))
                sample_key = f"{relative_metadata}#{object_id}"
                if self.split == "train":
                    if manifest_keys is not None:
                        if sample_key not in manifest_keys:
                            continue
                samples.append(
                    {
                        "metadata_path": metadata_path,
                        "metadata_relative": relative_metadata,
                        "object_index": object_index,
                        "sample_key": sample_key,
                        "item": item,
                    }
                )
        if (
            self.split == "train"
            and manifest_keys is None
            and self.train_fraction < 1.0
        ):
            selected_count = max(1, int(round(len(samples) * self.train_fraction)))
            samples = sorted(
                samples,
                key=lambda sample: (
                    _stable_rank(sample["sample_key"], self.train_seed),
                    sample["sample_key"],
                ),
            )[:selected_count]
        return samples

    @staticmethod
    def _resolve_data_path(root: Path, value, kind: str, sample_key: str) -> Path:
        if value is None or not str(value).strip():
            raise KeyError(f"RealVLG sample {sample_key} has no {kind} path")
        path = root / str(value).lstrip("/\\")
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing RealVLG {kind} for {sample_key}: {path}"
            )
        return path

    @staticmethod
    def _read_image(path: Path, flag, kind: str):
        value = cv2.imread(os.fspath(path), flag)
        if value is None:
            raise ValueError(f"OpenCV could not decode RealVLG {kind}: {path}")
        return value

    @staticmethod
    def _transform_matrix(image_size, input_size):
        original_h, original_w = image_size
        input_h, input_w = input_size
        scale = min(input_h / original_h, input_w / original_w)
        resized_h, resized_w = original_h * scale, original_w * scale
        bias_x = (input_w - resized_w) / 2.0
        bias_y = (input_h - resized_h) / 2.0
        source = np.array(
            [[0, 0], [original_w, 0], [0, original_h]], dtype=np.float32
        )
        destination = np.array(
            [
                [bias_x, bias_y],
                [resized_w + bias_x, bias_y],
                [bias_x, resized_h + bias_y],
            ],
            dtype=np.float32,
        )
        return (
            cv2.getAffineTransform(source, destination),
            cv2.getAffineTransform(destination, source),
            float(scale),
        )

    @staticmethod
    def _apply_affine(points, matrix):
        values = np.asarray(points, dtype=np.float32)
        ones = np.ones((*values.shape[:-1], 1), dtype=np.float32)
        homogeneous = np.concatenate([values, ones], axis=-1)
        return np.einsum("ij,...j->...i", matrix, homogeneous).astype(np.float32)

    @staticmethod
    def _transform_bbox(box_xyxy, matrix, input_size):
        box = np.asarray(box_xyxy, dtype=np.float32)
        if box.shape != (4,):
            raise ValueError(f"RealVLG bbox must be [x1,y1,x2,y2], got {box}")
        corners = np.array([[box[0], box[1]], [box[2], box[3]]], dtype=np.float32)
        corners = RealVLGDataset._apply_affine(corners, matrix)
        input_h, input_w = input_size
        first = np.maximum(corners[0], [0.0, 0.0])
        second = np.minimum(corners[1], [float(input_w), float(input_h)])
        return np.array([first[0], first[1], second[0], second[1]], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        item = record["item"]
        sample_key = record["sample_key"]
        required = ("image_path", "mask_path", "description", "bbox", "grasps")
        missing = [key for key in required if key not in item]
        if missing:
            raise KeyError(f"RealVLG sample {sample_key} is missing fields {missing}")

        image_path = self._resolve_data_path(
            self.root_dir, item["image_path"], "image", sample_key
        )
        mask_path = self._resolve_data_path(
            self.root_dir, item["mask_path"], "mask", sample_key
        )
        image = self._read_image(image_path, cv2.IMREAD_COLOR, "image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        mask_image = self._read_image(mask_path, cv2.IMREAD_GRAYSCALE, "mask")
        if mask_image.shape != (original_h, original_w):
            raise ValueError(
                f"RealVLG mask/image mismatch for {sample_key}: "
                f"mask={mask_image.shape}, image={(original_h, original_w)}"
            )
        original_mask = (mask_image > 128).astype(np.uint8)

        raw_grasps = np.asarray(item["grasps"], dtype=np.float32)
        if raw_grasps.ndim == 3 and raw_grasps.shape[1:] == (4, 2):
            raw_grasps = raw_grasps.reshape(-1, 8)
        if raw_grasps.ndim != 2 or raw_grasps.shape[1] != 8:
            raise ValueError(
                f"RealVLG grasps must have shape [N,8], got {raw_grasps.shape} "
                f"for {sample_key}"
            )
        raw_grasps = raw_grasps[np.isfinite(raw_grasps).all(axis=1)]
        if not len(raw_grasps):
            raise ValueError(f"RealVLG sample has no finite grasps: {sample_key}")
        original_rectangles = realvlg_points8_to_rectangles(raw_grasps)

        matrix, inverse, scale = self._transform_matrix(
            (original_h, original_w), self.input_size
        )
        input_points = self._apply_affine(raw_grasps.reshape(-1, 4, 2), matrix)
        input_rectangles = realvlg_points8_to_rectangles(input_points)
        raw_maps = self.grasp_transform.generate_masks(
            input_rectangles,
            consistent_owner=self.with_offset and self.offset_version == "v2",
        )
        angles = raw_maps["ang"].astype(np.float32) * np.pi / 180.0
        grasp_masks = {
            "qua": torch.from_numpy(
                raw_maps["qua"].astype(np.float32) / 255.0
            ),
            "sin": torch.from_numpy(np.sin(2.0 * angles)).float(),
            "cos": torch.from_numpy(np.cos(2.0 * angles)).float(),
            "wid": torch.from_numpy(
                raw_maps["wid"].astype(np.float32) / 255.0
            ),
        }
        if self.with_offset:
            offsets, weights = make_grasp_offset_targets_np(
                grasp_rectangles=input_rectangles,
                img_size_hw=self.input_size,
                offset_version=self.offset_version,
                offset_radius=self.offset_radius,
                offset_sigma=self.offset_sigma,
                target_stride=self.offset_target_stride,
                weight_floor=self.offset_weight_floor,
            )
            grasp_masks["off"] = torch.from_numpy(offsets).float()
            grasp_masks["off_w"] = torch.from_numpy(weights).float()

        input_image = cv2.warpAffine(
            image,
            matrix,
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        input_image = torch.from_numpy(input_image.transpose(2, 0, 1)).float()
        input_image.div_(255.0).sub_(self.mean).div_(self.std)
        input_mask = cv2.warpAffine(
            original_mask,
            matrix,
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        sentence = str(item["description"]).strip()
        if not sentence:
            raise ValueError(f"RealVLG description is empty: {sample_key}")
        word_vec = tokenize(sentence, self.word_length, True).squeeze(0).long()

        bbox_original = np.asarray(item.get("bbox", []), dtype=np.float32)
        if bbox_original.shape != (4,):
            raise ValueError(
                f"RealVLG bbox must be [x1,y1,x2,y2], got {bbox_original} "
                f"for {sample_key}"
            )
        return {
            "img": input_image,
            "depth": torch.zeros(1, *self.input_size, dtype=torch.float32),
            "mask": torch.from_numpy(input_mask).float(),
            "mask_original": original_mask,
            "grasp_masks": grasp_masks,
            "word_vec": word_vec,
            "grasps": original_rectangles,
            "grasps_points8": raw_grasps,
            "target": str(item.get("label", item.get("object_id", ""))),
            "sentence": sentence,
            "bbox": self._transform_bbox(
                bbox_original, matrix, self.input_size
            ),
            "bbox_original": bbox_original,
            "object_id": str(item.get("object_id", "")),
            "sample_id": sample_key,
            "scene_id": record["metadata_relative"].split("/", 1)[0],
            "frame_id": Path(record["metadata_relative"]).stem,
            "metadata_path": os.fspath(record["metadata_path"]),
            "inverse": inverse,
            "scale": scale,
            "ori_size": np.array([original_h, original_w], dtype=np.int32),
            "img_path": os.fspath(image_path),
            "mask_path": os.fspath(mask_path),
        }

    @staticmethod
    def collate_fn(batch):
        grasp_masks = {
            key: torch.stack([sample["grasp_masks"][key] for sample in batch])
            for key in ("qua", "sin", "cos", "wid")
        }
        for key in ("off", "off_w"):
            if all(key in sample["grasp_masks"] for sample in batch):
                grasp_masks[key] = torch.stack(
                    [sample["grasp_masks"][key] for sample in batch]
                )
        result = {
            "img": torch.stack([sample["img"] for sample in batch]),
            "depth": torch.stack([sample["depth"] for sample in batch]),
            "mask": torch.stack([sample["mask"] for sample in batch]),
            "grasp_masks": grasp_masks,
            "word_vec": torch.stack([sample["word_vec"] for sample in batch]),
        }
        list_keys = (
            "mask_original",
            "grasps",
            "grasps_points8",
            "target",
            "sentence",
            "bbox",
            "bbox_original",
            "object_id",
            "sample_id",
            "scene_id",
            "frame_id",
            "metadata_path",
            "inverse",
            "scale",
            "ori_size",
            "img_path",
            "mask_path",
        )
        result.update({key: [sample[key] for sample in batch] for key in list_keys})
        return result
