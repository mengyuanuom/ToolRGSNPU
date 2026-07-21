#!/usr/bin/env python3
"""Generate a compositional, language-driven Grasp-Tools dataset.

The generator keeps each rendered scene only once and stores multiple language
queries in the paired JSON file.  Every query contains a target_idx pointing to
the target object.  Geometry is transformed consistently; grasp height remains
20 pixels to match ToolRGS evaluation.

Typical usage from the ToolRGS repository root:

    python tools/dataset_converters/grasp_tools/augment.py

Quick validation run:

    python tools/dataset_converters/grasp_tools/augment.py \
        --out-dir /tmp/grasp_tools_v2_smoke --smoke-test --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.grasp_tool_language import (
    CANONICAL_CATEGORY_NAMES,
    CATEGORY_DESCRIPTION_VARIANTS,
    COMMAND_TEMPLATES,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

CATEGORY_ALIASES = {
    "box": "box",
    "plier": "pliers",
    "pliers": "pliers",
    "hex key": "t-hex key",
    "t hex key": "t-hex key",
    "t-handle hex key": "t-hex key",
    "l hex key": "l-hex key",
    "l-shaped hex key": "l-hex key",
}

HARD_NEGATIVE_GROUPS = (
    ("t-hex key", "l-hex key"),
    ("pliers", "crimp tool", "clamps"),
    ("tape", "tape measure", "spool"),
    ("marker", "screwdriver", "file"),
    ("screw", "nut", "clip"),
)

# Description wording is also split.  Evaluation therefore changes both the
# command prefix and the referring expression, rather than merely swapping the
# first verb in an otherwise identical sentence.
ABSOLUTE_DESCRIPTION_TEMPLATES = {
    "leftmost": {
        "train": ("the leftmost object", "the object on the far left", "the object farthest to the left"),
        "eval": ("the item at the left-hand edge", "the object positioned furthest left"),
    },
    "rightmost": {
        "train": ("the rightmost object", "the object on the far right", "the object farthest to the right"),
        "eval": ("the item at the right-hand edge", "the object positioned furthest right"),
    },
    "topmost": {
        "train": ("the topmost object", "the object at the top", "the highest object in the image"),
        "eval": ("the item at the upper edge", "the object positioned highest"),
    },
    "bottommost": {
        "train": ("the bottommost object", "the object at the bottom", "the lowest object in the image"),
        "eval": ("the item at the lower edge", "the object positioned lowest"),
    },
}

SAME_CATEGORY_DESCRIPTION_TEMPLATES = {
    "leftmost": {
        "train": ("the leftmost {category}", "the {category} on the far left"),
        "eval": ("the {category} at the left-hand edge", "the {category} positioned furthest left"),
    },
    "rightmost": {
        "train": ("the rightmost {category}", "the {category} on the far right"),
        "eval": ("the {category} at the right-hand edge", "the {category} positioned furthest right"),
    },
    "topmost": {
        "train": ("the topmost {category}", "the highest {category} in the image"),
        "eval": ("the {category} at the upper edge", "the {category} positioned highest"),
    },
    "bottommost": {
        "train": ("the bottommost {category}", "the lowest {category} in the image"),
        "eval": ("the {category} at the lower edge", "the {category} positioned lowest"),
    },
}

DIRECTION_DESCRIPTION_TEMPLATES = {
    "left": {
        "train": (
            "the object immediately to the left of the {reference}",
            "the object directly left of the {reference}",
        ),
        "eval": (
            "the object on the left-hand side of the {reference}",
            "the item next to the {reference} on its left",
        ),
    },
    "right": {
        "train": (
            "the object immediately to the right of the {reference}",
            "the object directly right of the {reference}",
        ),
        "eval": (
            "the object on the right-hand side of the {reference}",
            "the item next to the {reference} on its right",
        ),
    },
    "above": {
        "train": ("the object immediately above the {reference}", "the object directly above the {reference}"),
        "eval": ("the item just above the {reference}", "the object on the upper side of the {reference}"),
    },
    "below": {
        "train": ("the object immediately below the {reference}", "the object directly below the {reference}"),
        "eval": ("the item just below the {reference}", "the object on the lower side of the {reference}"),
    },
}

NEAREST_DESCRIPTION_TEMPLATES = {
    "train": ("the object closest to the {reference}", "the object nearest to the {reference}"),
    "eval": ("the item right next to the {reference}", "the object at the shortest distance from the {reference}"),
}

FARTHEST_DESCRIPTION_TEMPLATES = {
    "train": ("the object farthest from the {reference}", "the object most distant from the {reference}"),
    "eval": ("the item furthest away from the {reference}", "the object at the greatest distance from the {reference}"),
}

BETWEEN_DESCRIPTION_TEMPLATES = {
    "train": (
        "the object between the {first} and the {second}",
        "the object in the middle of the {first} and the {second}",
    ),
    "eval": (
        "the item positioned midway between the {first} and the {second}",
        "the object lying in between the {first} and the {second}",
    ),
}


def format_description_variants(
    templates: Dict[str, Sequence[str]], **values: str
) -> Dict[str, List[str]]:
    return {
        split: [template.format(**values) for template in split_templates]
        for split, split_templates in templates.items()
    }


@dataclass(frozen=True)
class SourceObject:
    source_id: str
    image_path: Path
    object_index: int
    category_key: str
    category_name: str
    mask: Tuple[Tuple[float, float], ...]
    grasps: Tuple[Tuple[Tuple[float, float], ...], ...]


@dataclass
class PreparedObject:
    source: SourceObject
    rgba: np.ndarray
    polygon_local: np.ndarray
    grasps_local: List[np.ndarray]


@dataclass
class TransformedObject:
    source: SourceObject
    rgba: np.ndarray
    polygon: np.ndarray
    grasps: List[np.ndarray]
    scale: float
    angle_deg: float


@dataclass
class GeneratorConfig:
    src_dir: str
    background_dir: str
    out_dir: str
    train_scenes: int
    val_scenes: int
    test_scenes: int
    objects_min: int
    objects_max: int
    queries_min: int
    queries_max: int
    max_query_difficulty: int
    language_templates: str
    category_vocabulary: str
    scales: Tuple[float, ...]
    angle_bins: int
    same_category_probability: float
    hard_negative_probability: float
    placement_attempts: int
    scene_attempts: int
    border_margin: int
    relation_margin: float
    nearest_ratio: float
    grasp_height: float
    brightness_jitter: float
    contrast_jitter: float
    saturation_jitter: float
    feather_radius: float
    seed: int
    image_ext: str
    jpeg_quality: int
    preview_count: int
    overwrite: bool


def parse_scales(value: str) -> Tuple[float, ...]:
    try:
        scales = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid scales: {value}") from exc
    if not scales or any(scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("All scales must be positive")
    return scales


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a multi-object, multi-query Grasp-Tools dataset."
    )
    parser.add_argument("--src-dir", default="assets/grasp_tools/graspall")
    parser.add_argument("--background-dir", default="assets/grasp_tools/backgrounds")
    parser.add_argument("--out-dir", default="datasets/grasp-tools/aug_graspall_v2")
    parser.add_argument("--train-scenes", type=int, default=3000)
    parser.add_argument("--val-scenes", type=int, default=500)
    parser.add_argument("--test-scenes", type=int, default=1000)
    parser.add_argument("--objects-min", type=int, default=2)
    parser.add_argument("--objects-max", type=int, default=3)
    parser.add_argument("--queries-min", type=int, default=2)
    parser.add_argument("--queries-max", type=int, default=4)
    parser.add_argument(
        "--max-query-difficulty",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help=(
            "Keep queries up to this difficulty: 1=category, "
            "2=category and absolute location, 3=single-reference relations, "
            "4=all queries including between relations."
        ),
    )
    parser.add_argument(
        "--language-templates",
        choices=("heldout", "shared"),
        default="shared",
        help=(
            "Use held-out wording for validation/test, or share the training "
            "wording across all splits."
        ),
    )
    parser.add_argument(
        "--category-vocabulary",
        choices=("canonical", "expanded"),
        default="expanded",
        help=(
            "Use only canonical category names, or sample aliases and common "
            "near-synonyms while preserving canonical labels."
        ),
    )
    parser.add_argument(
        "--scales", type=parse_scales, default=parse_scales("0.9,1.0,1.15,1.3")
    )
    parser.add_argument("--angle-bins", type=int, default=8)
    parser.add_argument("--same-category-probability", type=float, default=0.0)
    parser.add_argument("--hard-negative-probability", type=float, default=0.0)
    parser.add_argument("--placement-attempts", type=int, default=200)
    parser.add_argument("--scene-attempts", type=int, default=30)
    parser.add_argument("--border-margin", type=int, default=4)
    parser.add_argument("--relation-margin", type=float, default=35.0)
    parser.add_argument("--nearest-ratio", type=float, default=1.20)
    parser.add_argument("--grasp-height", type=float, default=20.0)
    parser.add_argument("--brightness-jitter", type=float, default=0.05)
    parser.add_argument("--contrast-jitter", type=float, default=0.05)
    parser.add_argument("--saturation-jitter", type=float, default=0.05)
    parser.add_argument("--feather-radius", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--image-ext", choices=("png", "jpg"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate 4 train, 2 val, and 2 test scenes.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> GeneratorConfig:
    if args.smoke_test:
        args.train_scenes, args.val_scenes, args.test_scenes = 4, 2, 2
        args.preview_count = min(args.preview_count, 8)

    if args.objects_min < 2 or args.objects_max < args.objects_min:
        raise ValueError("objects-min must be >=2 and <= objects-max")
    if args.queries_min < 1 or args.queries_max < args.queries_min:
        raise ValueError("queries-min must be >=1 and <= queries-max")
    if not 1 <= args.max_query_difficulty <= 4:
        raise ValueError("max-query-difficulty must be in [1, 4]")
    if args.language_templates not in {"heldout", "shared"}:
        raise ValueError("language-templates must be 'heldout' or 'shared'")
    if args.category_vocabulary not in {"canonical", "expanded"}:
        raise ValueError("category-vocabulary must be 'canonical' or 'expanded'")
    if args.angle_bins < 1:
        raise ValueError("angle-bins must be positive")
    for name in ("same_category_probability", "hard_negative_probability"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if args.grasp_height <= 0:
        raise ValueError("grasp-height must be positive")

    return GeneratorConfig(
        src_dir=args.src_dir,
        background_dir=args.background_dir,
        out_dir=args.out_dir,
        train_scenes=args.train_scenes,
        val_scenes=args.val_scenes,
        test_scenes=args.test_scenes,
        objects_min=args.objects_min,
        objects_max=args.objects_max,
        queries_min=args.queries_min,
        queries_max=args.queries_max,
        max_query_difficulty=args.max_query_difficulty,
        language_templates=args.language_templates,
        category_vocabulary=args.category_vocabulary,
        scales=tuple(args.scales),
        angle_bins=args.angle_bins,
        same_category_probability=args.same_category_probability,
        hard_negative_probability=args.hard_negative_probability,
        placement_attempts=args.placement_attempts,
        scene_attempts=args.scene_attempts,
        border_margin=args.border_margin,
        relation_margin=args.relation_margin,
        nearest_ratio=args.nearest_ratio,
        grasp_height=args.grasp_height,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
        saturation_jitter=args.saturation_jitter,
        feather_radius=args.feather_radius,
        seed=args.seed,
        image_ext=args.image_ext,
        jpeg_quality=args.jpeg_quality,
        preview_count=args.preview_count,
        overwrite=args.overwrite,
    )


def canonicalize_category(value: str) -> Tuple[str, str]:
    key = (value or "").strip().lower().replace("_", " ")
    key = key.replace("—", "-").replace("–", "-")
    key = " ".join(key.split())
    key = CATEGORY_ALIASES.get(key, key)
    if key not in CANONICAL_CATEGORY_NAMES:
        raise ValueError(f"Unknown category: {value!r}")
    return key, CANONICAL_CATEGORY_NAMES[key]


def list_images(directory: Path) -> List[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_source_objects(src_dir: Path) -> Tuple[List[SourceObject], List[str]]:
    objects: List[SourceObject] = []
    warnings: List[str] = []
    for image_path in sorted(
        path for path in src_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            warnings.append(f"missing JSON for {image_path.name}")
            continue
        try:
            annotation = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"invalid JSON {json_path.name}: {exc}")
            continue

        for object_index, obj in enumerate(annotation.get("objects", [])):
            try:
                category_key, category_name = canonicalize_category(obj.get("category", ""))
            except ValueError as exc:
                warnings.append(f"{json_path.name} object {object_index}: {exc}")
                continue
            mask = obj.get("mask") or []
            grasps = [grasp for grasp in (obj.get("grasps") or []) if len(grasp) == 4]
            if len(mask) < 3:
                warnings.append(f"{json_path.name} object {object_index}: empty/invalid mask")
                continue
            if not grasps:
                warnings.append(f"{json_path.name} object {object_index}: no valid grasps")
                continue
            try:
                mask_tuple = tuple((float(point[0]), float(point[1])) for point in mask)
                grasp_tuple = tuple(
                    tuple((float(point[0]), float(point[1])) for point in grasp)
                    for grasp in grasps
                )
            except (TypeError, ValueError, IndexError) as exc:
                warnings.append(f"{json_path.name} object {object_index}: malformed coordinates: {exc}")
                continue
            objects.append(
                SourceObject(
                    source_id=f"{image_path.stem}:{object_index}",
                    image_path=image_path,
                    object_index=object_index,
                    category_key=category_key,
                    category_name=category_name,
                    mask=mask_tuple,
                    grasps=grasp_tuple,
                )
            )
    return objects, warnings


def polygon_mask(points: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) >= 3:
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


def prepare_source_object(source: SourceObject) -> PreparedObject:
    image = cv2.imread(str(source.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read source image: {source.image_path}")
    polygon = np.asarray(source.mask, dtype=np.float32)
    height, width = image.shape[:2]
    x1 = max(0, int(math.floor(float(polygon[:, 0].min()))))
    y1 = max(0, int(math.floor(float(polygon[:, 1].min()))))
    x2 = min(width - 1, int(math.ceil(float(polygon[:, 0].max()))))
    y2 = min(height - 1, int(math.ceil(float(polygon[:, 1].max()))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate mask crop for {source.source_id}")

    crop = image[y1:y2 + 1, x1:x2 + 1].copy()
    polygon_local = polygon - np.asarray([x1, y1], dtype=np.float32)
    alpha = polygon_mask(polygon_local, crop.shape[0], crop.shape[1])
    rgba = np.dstack((crop, alpha))
    grasps_local = [
        np.asarray(grasp, dtype=np.float32) - np.asarray([x1, y1], dtype=np.float32)
        for grasp in source.grasps
    ]
    return PreparedObject(source, rgba, polygon_local, grasps_local)


def apply_photometric_jitter(
    rgba: np.ndarray,
    rng: random.Random,
    brightness: float,
    contrast: float,
    saturation: float,
) -> np.ndarray:
    result = rgba.copy()
    rgb = result[:, :, :3].astype(np.float32)
    if brightness > 0:
        rgb *= rng.uniform(1.0 - brightness, 1.0 + brightness)
    if contrast > 0:
        factor = rng.uniform(1.0 - contrast, 1.0 + contrast)
        mean = rgb.mean(axis=(0, 1), keepdims=True)
        rgb = (rgb - mean) * factor + mean
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if saturation > 0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(1.0 - saturation, 1.0 + saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    result[:, :, :3] = rgb
    return result


def affine_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    return np.hstack((points.astype(np.float32), ones)) @ matrix.T


def transform_object(
    prepared: PreparedObject,
    scale: float,
    angle_deg: float,
    rng: random.Random,
    config: GeneratorConfig,
) -> TransformedObject:
    rgba = apply_photometric_jitter(
        prepared.rgba,
        rng,
        config.brightness_jitter,
        config.contrast_jitter,
        config.saturation_jitter,
    )
    height, width = rgba.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, scale)
    crop_corners = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    warped_corners = affine_points(crop_corners, matrix)
    min_xy = warped_corners.min(axis=0)
    max_xy = warped_corners.max(axis=0)
    out_width = max(1, int(math.ceil(float(max_xy[0] - min_xy[0]))))
    out_height = max(1, int(math.ceil(float(max_xy[1] - min_xy[1]))))
    matrix[0, 2] -= float(min_xy[0])
    matrix[1, 2] -= float(min_xy[1])
    warped = cv2.warpAffine(
        rgba,
        matrix,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    if config.feather_radius > 0:
        sigma = float(config.feather_radius)
        warped[:, :, 3] = cv2.GaussianBlur(warped[:, :, 3], (0, 0), sigmaX=sigma)
    polygon = affine_points(prepared.polygon_local, matrix)
    grasps = [affine_points(grasp, matrix) for grasp in prepared.grasps_local]
    return TransformedObject(
        source=prepared.source,
        rgba=warped,
        polygon=polygon,
        grasps=grasps,
        scale=scale,
        angle_deg=angle_deg,
    )


def decode_grasp(points: np.ndarray) -> Tuple[float, float, float, float]:
    center = points.mean(axis=0)
    delta = points[1] - points[0]
    width = float(np.linalg.norm(delta))
    angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    return float(center[0]), float(center[1]), angle, width


def fixed_height_grasp(
    center_x: float,
    center_y: float,
    angle_deg: float,
    width: float,
    height: float,
) -> np.ndarray:
    radians = math.radians(angle_deg)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    half_width, half_height = width / 2.0, height / 2.0
    width_vec = np.asarray([cos_a * half_width, sin_a * half_width])
    height_vec = np.asarray([-sin_a * half_height, cos_a * half_height])
    center = np.asarray([center_x, center_y])
    return np.asarray(
        [
            center - width_vec - height_vec,
            center + width_vec - height_vec,
            center + width_vec + height_vec,
            center - width_vec + height_vec,
        ],
        dtype=np.float32,
    )


def points_inside(points: np.ndarray, width: int, height: int) -> bool:
    return bool(
        np.all(points[:, 0] >= 0)
        and np.all(points[:, 0] < width)
        and np.all(points[:, 1] >= 0)
        and np.all(points[:, 1] < height)
    )


def paste_object(
    canvas: np.ndarray,
    occupancy: np.ndarray,
    transformed: TransformedObject,
    rng: random.Random,
    config: GeneratorConfig,
) -> Optional[Dict[str, Any]]:
    canvas_height, canvas_width = canvas.shape[:2]
    obj_height, obj_width = transformed.rgba.shape[:2]
    min_x = config.border_margin
    min_y = config.border_margin
    max_x = canvas_width - obj_width - config.border_margin
    max_y = canvas_height - obj_height - config.border_margin
    if max_x < min_x or max_y < min_y:
        return None

    alpha_hard = transformed.rgba[:, :, 3] >= 8
    if not np.any(alpha_hard):
        return None

    for _ in range(config.placement_attempts):
        x = rng.randint(min_x, max_x)
        y = rng.randint(min_y, max_y)
        occupied_patch = occupancy[y:y + obj_height, x:x + obj_width]
        if np.any(alpha_hard & occupied_patch):
            continue

        translation = np.asarray([x, y], dtype=np.float32)
        polygon = transformed.polygon + translation
        transformed_grasps: List[np.ndarray] = []
        for grasp in transformed.grasps:
            grasp_canvas = grasp + translation
            center_x, center_y, angle, grasp_width = decode_grasp(grasp_canvas)
            rebuilt = fixed_height_grasp(
                center_x,
                center_y,
                angle,
                grasp_width,
                config.grasp_height,
            )
            if points_inside(rebuilt, canvas_width, canvas_height):
                transformed_grasps.append(rebuilt)
        if not transformed_grasps:
            continue

        alpha = transformed.rgba[:, :, 3:4].astype(np.float32) / 255.0
        foreground = transformed.rgba[:, :, :3].astype(np.float32)
        background = canvas[y:y + obj_height, x:x + obj_width].astype(np.float32)
        blended = foreground * alpha + background * (1.0 - alpha)
        canvas[y:y + obj_height, x:x + obj_width] = np.clip(blended, 0, 255).astype(np.uint8)
        occupancy[y:y + obj_height, x:x + obj_width] |= alpha_hard

        polygon[:, 0] = np.clip(polygon[:, 0], 0, canvas_width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, canvas_height - 1)
        x1 = int(math.floor(float(polygon[:, 0].min())))
        y1 = int(math.floor(float(polygon[:, 1].min())))
        x2 = int(math.ceil(float(polygon[:, 0].max())))
        y2 = int(math.ceil(float(polygon[:, 1].max())))
        return {
            "object_id": -1,
            "source_id": transformed.source.source_id,
            "category": transformed.source.category_name,
            "mask": polygon.astype(float).tolist(),
            "bbox": [x1, y1, x2, y2],
            "grasps": [grasp.astype(float).tolist() for grasp in transformed_grasps],
            "transform": {
                "scale": round(float(transformed.scale), 6),
                "rotation_deg": round(float(transformed.angle_deg % 360.0), 6),
                "translation": [int(x), int(y)],
            },
        }
    return None


class BalancedCategorySampler:
    def __init__(self, categories: Sequence[str], rng: random.Random):
        self.categories = list(categories)
        self.rng = rng
        self.queue: Deque[str] = deque()

    def next(self) -> str:
        if not self.queue:
            values = self.categories.copy()
            self.rng.shuffle(values)
            self.queue.extend(values)
        return self.queue.popleft()


class BalancedTransformSampler:
    def __init__(
        self,
        objects_by_category: Dict[str, List[SourceObject]],
        scales: Sequence[float],
        angle_bins: int,
        rng: random.Random,
    ):
        self.objects_by_category = objects_by_category
        self.scales = tuple(scales)
        self.angle_bins = int(angle_bins)
        self.rng = rng
        self.queues: Dict[str, Deque[Tuple[SourceObject, float, int]]] = {}

    def _refill(self, category: str) -> None:
        tasks = [
            (source, scale, angle_bin)
            for source in self.objects_by_category[category]
            for scale in self.scales
            for angle_bin in range(self.angle_bins)
        ]
        self.rng.shuffle(tasks)
        self.queues[category] = deque(tasks)

    def next(self, category: str) -> Tuple[SourceObject, float, float]:
        if category not in self.queues or not self.queues[category]:
            self._refill(category)
        source, scale, angle_bin = self.queues[category].popleft()
        bin_width = 360.0 / self.angle_bins
        center = angle_bin * bin_width
        angle = (center + self.rng.uniform(-bin_width / 2.0, bin_width / 2.0)) % 360.0
        return source, scale, angle


def hard_neighbors(category: str, available: Iterable[str]) -> List[str]:
    available_set = set(available)
    for group in HARD_NEGATIVE_GROUPS:
        if category in group:
            return [item for item in group if item != category and item in available_set]
    return []


def choose_scene_categories(
    count: int,
    category_sampler: BalancedCategorySampler,
    available: Sequence[str],
    rng: random.Random,
    same_category_probability: float,
    hard_negative_probability: float,
) -> List[str]:
    anchor = category_sampler.next()
    selected = [anchor]
    if count > 1 and rng.random() < same_category_probability:
        selected.append(anchor)
    if len(selected) < count and rng.random() < hard_negative_probability:
        candidates = hard_neighbors(anchor, available)
        if candidates:
            selected.append(rng.choice(candidates))
    fill = list(available)
    rng.shuffle(fill)
    for category in fill:
        if len(selected) >= count:
            break
        if category not in selected:
            selected.append(category)
    while len(selected) < count:
        selected.append(rng.choice(list(available)))
    rng.shuffle(selected)
    return selected[:count]


def object_centers(objects: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [(obj["bbox"][0] + obj["bbox"][2]) / 2.0,
             (obj["bbox"][1] + obj["bbox"][3]) / 2.0]
            for obj in objects
        ],
        dtype=np.float32,
    )


def category_key_from_object(obj: Dict[str, Any]) -> str:
    return canonicalize_category(obj["category"])[0]


def logical_candidates(
    objects: Sequence[Dict[str, Any]],
    canvas_width: int,
    canvas_height: int,
    margin: float,
    nearest_ratio: float,
) -> List[Dict[str, Any]]:
    centers = object_centers(objects)
    categories = [category_key_from_object(obj) for obj in objects]
    category_counts = Counter(categories)
    candidates: List[Dict[str, Any]] = []

    for index, category in enumerate(categories):
        if category_counts[category] == 1:
            candidates.append({
                "target_idx": index,
                "type": "category",
                "difficulty": 1,
                "category_key": category,
                "description": f"the {CANONICAL_CATEGORY_NAMES[category]}",
                "program": [{"op": "filter_category", "value": category}, {"op": "unique"}],
            })

    axes = (
        (0, True, "leftmost"),
        (0, False, "rightmost"),
        (1, True, "topmost"),
        (1, False, "bottommost"),
    )
    for axis, ascending, relation in axes:
        order = np.argsort(centers[:, axis])
        if not ascending:
            order = order[::-1]
        if len(order) >= 2 and abs(float(centers[order[1], axis] - centers[order[0], axis])) >= margin:
            target = int(order[0])
            candidates.append({
                "target_idx": target,
                "type": "absolute_location",
                "difficulty": 2,
                "descriptions": format_description_variants(
                    ABSOLUTE_DESCRIPTION_TEMPLATES[relation]
                ),
                "program": [{"op": "scene"}, {"op": relation}, {"op": "unique"}],
            })

    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for index, category in enumerate(categories):
        grouped_indices[category].append(index)
    for category, indices in grouped_indices.items():
        if len(indices) < 2:
            continue
        subset = centers[indices]
        for axis, ascending, relation in axes:
            order = np.argsort(subset[:, axis])
            if not ascending:
                order = order[::-1]
            first = indices[int(order[0])]
            second = indices[int(order[1])]
            if abs(float(centers[first, axis] - centers[second, axis])) < margin:
                continue
            candidates.append({
                "target_idx": first,
                "type": "same_category_location",
                "difficulty": 3,
                "descriptions": format_description_variants(
                    SAME_CATEGORY_DESCRIPTION_TEMPLATES[relation],
                    category=CANONICAL_CATEGORY_NAMES[category],
                ),
                "program": [
                    {"op": "filter_category", "value": category},
                    {"op": relation},
                    {"op": "unique"},
                ],
            })

    unique_reference_indices = [
        index for index, category in enumerate(categories) if category_counts[category] == 1
    ]
    for reference in unique_reference_indices:
        other_indices = [index for index in range(len(objects)) if index != reference]
        distances = sorted(
            ((float(np.linalg.norm(centers[index] - centers[reference])), index)
             for index in other_indices),
            key=lambda item: item[0],
        )
        if distances:
            unique_nearest = len(distances) == 1 or distances[1][0] >= distances[0][0] * nearest_ratio
            if unique_nearest:
                target = distances[0][1]
                reference_name = CANONICAL_CATEGORY_NAMES[categories[reference]]
                candidates.append({
                    "target_idx": target,
                    "type": "nearest_relation",
                    "difficulty": 3,
                    "descriptions": format_description_variants(
                        NEAREST_DESCRIPTION_TEMPLATES, reference=reference_name
                    ),
                    "reference_indices": [reference],
                    "program": [
                        {"op": "filter_category", "value": categories[reference]},
                        {"op": "unique"},
                        {"op": "relate", "value": "nearest"},
                        {"op": "unique"},
                    ],
                })

            farthest = list(reversed(distances))
            unique_farthest = (
                len(farthest) == 1
                or farthest[0][0] >= farthest[1][0] * nearest_ratio
            )
            if unique_farthest:
                target = farthest[0][1]
                reference_name = CANONICAL_CATEGORY_NAMES[categories[reference]]
                candidates.append({
                    "target_idx": target,
                    "type": "farthest_relation",
                    "difficulty": 3,
                    "descriptions": format_description_variants(
                        FARTHEST_DESCRIPTION_TEMPLATES, reference=reference_name
                    ),
                    "reference_indices": [reference],
                    "program": [
                        {"op": "filter_category", "value": categories[reference]},
                        {"op": "unique"},
                        {"op": "relate", "value": "farthest"},
                        {"op": "unique"},
                    ],
                })

        directions = (
            ("left", np.asarray([-1.0, 0.0]), 0),
            ("right", np.asarray([1.0, 0.0]), 0),
            ("above", np.asarray([0.0, -1.0]), 1),
            ("below", np.asarray([0.0, 1.0]), 1),
        )
        for name, direction, primary_axis in directions:
            directional: List[Tuple[float, int]] = []
            for target in other_indices:
                delta = centers[target] - centers[reference]
                projection = float(np.dot(delta, direction))
                orthogonal = abs(float(delta[1 - primary_axis]))
                alignment_limit = 0.28 * (canvas_height if primary_axis == 0 else canvas_width)
                if projection >= margin and orthogonal <= alignment_limit:
                    score = projection + 0.35 * orthogonal
                    directional.append((score, target))
            directional.sort(key=lambda item: item[0])
            if not directional:
                continue
            if len(directional) > 1 and directional[1][0] < directional[0][0] * nearest_ratio:
                continue
            target = directional[0][1]
            reference_name = CANONICAL_CATEGORY_NAMES[categories[reference]]
            candidates.append({
                "target_idx": target,
                "type": "direction_relation",
                "difficulty": 3,
                "descriptions": format_description_variants(
                    DIRECTION_DESCRIPTION_TEMPLATES[name], reference=reference_name
                ),
                "reference_indices": [reference],
                "program": [
                    {"op": "filter_category", "value": categories[reference]},
                    {"op": "unique"},
                    {"op": "relate", "value": name},
                    {"op": "nearest_in_direction"},
                    {"op": "unique"},
                ],
            })

    for left_pos, first_ref in enumerate(unique_reference_indices):
        for second_ref in unique_reference_indices[left_pos + 1:]:
            segment = centers[second_ref] - centers[first_ref]
            segment_sq = float(np.dot(segment, segment))
            if segment_sq < (2.0 * margin) ** 2:
                continue
            between_scores: List[Tuple[float, int]] = []
            for target in range(len(objects)):
                if target in (first_ref, second_ref):
                    continue
                relative = centers[target] - centers[first_ref]
                projection = float(np.dot(relative, segment) / segment_sq)
                if not 0.20 <= projection <= 0.80:
                    continue
                closest = centers[first_ref] + projection * segment
                perpendicular = float(np.linalg.norm(centers[target] - closest))
                if perpendicular <= max(margin, 0.08 * math.hypot(canvas_width, canvas_height)):
                    between_scores.append((perpendicular, target))
            between_scores.sort(key=lambda item: item[0])
            if len(between_scores) != 1:
                continue
            target = between_scores[0][1]
            first_name = CANONICAL_CATEGORY_NAMES[categories[first_ref]]
            second_name = CANONICAL_CATEGORY_NAMES[categories[second_ref]]
            candidates.append({
                "target_idx": target,
                "type": "between_relation",
                "difficulty": 4,
                "descriptions": format_description_variants(
                    BETWEEN_DESCRIPTION_TEMPLATES,
                    first=first_name,
                    second=second_name,
                ),
                "reference_indices": [first_ref, second_ref],
                "program": [
                    {"op": "filter_category", "value": categories[first_ref]},
                    {"op": "filter_category", "value": categories[second_ref]},
                    {"op": "relate", "value": "between"},
                    {"op": "unique"},
                ],
            })
    return candidates


def render_queries(
    candidates: Sequence[Dict[str, Any]],
    split: str,
    scene_id: str,
    minimum: int,
    maximum: int,
    max_difficulty: int,
    language_templates: str,
    category_vocabulary: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if int(candidate["difficulty"]) <= max_difficulty:
            by_type[candidate["type"]].append(candidate)
    for values in by_type.values():
        rng.shuffle(values)

    type_priority = [
        "same_category_location",
        "between_relation",
        "direction_relation",
        "nearest_relation",
        "farthest_relation",
        "absolute_location",
        "category",
    ]
    selected: List[Dict[str, Any]] = []
    while len(selected) < maximum:
        added = False
        for query_type in type_priority:
            if by_type[query_type] and len(selected) < maximum:
                selected.append(by_type[query_type].pop())
                added = True
        if not added:
            break

    if len(selected) < minimum:
        return []

    template_split = (
        "train"
        if split == "train" or language_templates == "shared"
        else "eval"
    )
    template_pool = COMMAND_TEMPLATES[template_split]
    queries: List[Dict[str, Any]] = []
    used_texts = set()
    for candidate in selected:
        description_split = template_split
        description_pool = candidate.get("descriptions", {}).get(description_split)
        category_term = None
        if candidate["type"] == "category":
            category_key = candidate["category_key"]
            if category_vocabulary == "expanded":
                category_term = rng.choice(CATEGORY_DESCRIPTION_VARIANTS[category_key])
            else:
                category_term = CANONICAL_CATEGORY_NAMES[category_key]
            description = f"the {category_term}"
        elif description_pool:
            description = rng.choice(list(description_pool))
        else:
            description = candidate["description"]
        templates = list(template_pool)
        rng.shuffle(templates)
        text = ""
        for template in templates:
            proposal = template.format(description=description)
            if proposal not in used_texts:
                text = proposal
                break
        if not text:
            continue
        used_texts.add(text)
        query = {
            "query_id": f"{scene_id}_q{len(queries):02d}",
            "text": text,
            "target_idx": int(candidate["target_idx"]),
            "type": candidate["type"],
            "difficulty": int(candidate["difficulty"]),
            "program": candidate.get("program", []),
        }
        if candidate.get("reference_indices"):
            query["reference_indices"] = [int(i) for i in candidate["reference_indices"]]
        if category_term is not None:
            query["category_term"] = category_term
            if category_vocabulary == "expanded":
                query["prompt_cycle"] = "category_v1"
        queries.append(query)
    return queries


def split_backgrounds(paths: Sequence[Path], seed: int) -> Dict[str, List[Path]]:
    if len(paths) < 3:
        raise ValueError("At least three background images are required")
    values = list(paths)
    random.Random(seed).shuffle(values)
    total = len(values)
    val_count = max(1, int(round(total * 0.13)))
    test_count = max(1, int(round(total * 0.13)))
    train_count = total - val_count - test_count
    if train_count < 1:
        train_count = 1
        if val_count > test_count:
            val_count -= 1
        else:
            test_count -= 1
    return {
        "train": values[:train_count],
        "val": values[train_count:train_count + val_count],
        "test": values[train_count + val_count:],
    }


def prepare_output_directory(config: GeneratorConfig, src_dir: Path, background_dir: Path) -> Path:
    out_dir = Path(config.out_dir).expanduser().resolve()
    protected = {src_dir.resolve(), background_dir.resolve(), Path.cwd().resolve()}
    if out_dir in protected:
        raise ValueError(f"Refusing to use protected directory as output: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {out_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "_preview").mkdir(parents=True, exist_ok=True)
    return out_dir


def save_image(path: Path, image: np.ndarray, config: GeneratorConfig) -> None:
    parameters: List[int] = []
    if config.image_ext == "jpg":
        parameters = [cv2.IMWRITE_JPEG_QUALITY, int(config.jpeg_quality)]
    if not cv2.imwrite(str(path), image, parameters):
        raise IOError(f"Failed to write image: {path}")


def draw_preview(image: np.ndarray, annotation: Dict[str, Any]) -> np.ndarray:
    preview = image.copy()
    palette = ((0, 255, 0), (255, 0, 255), (0, 200, 255), (255, 180, 0), (180, 255, 0))
    for index, obj in enumerate(annotation["objects"]):
        color = palette[index % len(palette)]
        polygon = np.rint(np.asarray(obj["mask"], dtype=np.float32)).astype(np.int32)
        cv2.polylines(preview, [polygon.reshape(-1, 1, 2)], True, color, 2)
        for grasp in obj["grasps"][::max(1, len(obj["grasps"]) // 8)]:
            points = np.rint(np.asarray(grasp, dtype=np.float32)).astype(np.int32)
            cv2.polylines(preview, [points.reshape(-1, 1, 2)], True, color, 1)
        x1, y1, _, _ = obj["bbox"]
        cv2.putText(
            preview,
            f"{index}:{obj['category']}",
            (max(0, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return preview


def validate_annotation(annotation: Dict[str, Any], width: int, height: int) -> None:
    objects = annotation.get("objects") or []
    queries = annotation.get("queries") or []
    if len(objects) < 2:
        raise ValueError("Generated scene has fewer than two objects")
    if not queries:
        raise ValueError("Generated scene has no queries")
    for index, obj in enumerate(objects):
        if obj.get("object_id") != index:
            raise ValueError("object_id does not match object list index")
        canonicalize_category(obj.get("category", ""))
        polygon = np.asarray(obj.get("mask", []), dtype=np.float32)
        if polygon.shape[0] < 3 or not points_inside(polygon, width, height):
            raise ValueError(f"Object {index} has invalid/out-of-bounds mask")
        if not obj.get("grasps"):
            raise ValueError(f"Object {index} has no grasps")
        for grasp in obj["grasps"]:
            points = np.asarray(grasp, dtype=np.float32)
            if points.shape != (4, 2) or not points_inside(points, width, height):
                raise ValueError(f"Object {index} has invalid/out-of-bounds grasp")
    for query in queries:
        target_idx = int(query.get("target_idx", -1))
        if not 0 <= target_idx < len(objects):
            raise ValueError(f"Invalid query target_idx: {target_idx}")
        if not str(query.get("text", "")).strip():
            raise ValueError("Query has empty text")


def build_scene(
    split: str,
    scene_number: int,
    background_paths: Sequence[Path],
    objects_by_category: Dict[str, List[SourceObject]],
    prepared_cache: Dict[str, PreparedObject],
    category_sampler: BalancedCategorySampler,
    transform_sampler: BalancedTransformSampler,
    rng: random.Random,
    config: GeneratorConfig,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    available_categories = sorted(objects_by_category)
    scene_id = f"{split}_scene_{scene_number:06d}"

    for scene_attempt in range(config.scene_attempts):
        background_path = background_paths[(scene_number + scene_attempt) % len(background_paths)]
        canvas = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if canvas is None:
            raise FileNotFoundError(f"Cannot read background: {background_path}")
        occupancy = np.zeros(canvas.shape[:2], dtype=bool)
        requested_count = rng.randint(config.objects_min, config.objects_max)
        category_plan = choose_scene_categories(
            requested_count,
            category_sampler,
            available_categories,
            rng,
            config.same_category_probability,
            config.hard_negative_probability,
        )

        placed_objects: List[Dict[str, Any]] = []
        for category in category_plan:
            placed: Optional[Dict[str, Any]] = None
            for _ in range(12):
                source, scale, angle = transform_sampler.next(category)
                if source.source_id not in prepared_cache:
                    prepared_cache[source.source_id] = prepare_source_object(source)
                transformed = transform_object(
                    prepared_cache[source.source_id], scale, angle, rng, config
                )
                placed = paste_object(canvas, occupancy, transformed, rng, config)
                if placed is not None:
                    placed["object_id"] = len(placed_objects)
                    placed_objects.append(placed)
                    break
            if placed is None:
                continue

        if len(placed_objects) < config.objects_min:
            continue

        height, width = canvas.shape[:2]
        candidates = logical_candidates(
            placed_objects,
            width,
            height,
            config.relation_margin,
            config.nearest_ratio,
        )
        queries = render_queries(
            candidates,
            split,
            scene_id,
            config.queries_min,
            config.queries_max,
            config.max_query_difficulty,
            config.language_templates,
            config.category_vocabulary,
            rng,
        )
        if len(queries) < config.queries_min:
            continue

        annotation = {
            "schema_version": "2.0",
            "split": split,
            "scene_id": scene_id,
            "image_filename": f"{scene_id}.{config.image_ext}",
            "background_source": background_path.name,
            "image_size": [int(width), int(height)],
            "objects": placed_objects,
            "queries": queries,
        }
        validate_annotation(annotation, width, height)
        return canvas, annotation
    return None


def write_readme(out_dir: Path, config: GeneratorConfig) -> None:
    text = f"""Grasp-Tools compositional augmentation v2

Schema:
  Each split contains one image and one JSON per rendered scene.
  JSON objects contain masks and grasp rectangles.
  JSON queries contain text, target_idx, type, difficulty, and a symbolic program.
  The same image is intentionally shared by multiple language queries.

Important:
  - grasp rectangle height is fixed at {config.grasp_height:g} pixels.
  - maximum query difficulty is {config.max_query_difficulty}.
  - language template protocol is {config.language_templates}.
  - category vocabulary is {config.category_vocabulary}.
  - train/val/test use disjoint background image files.
  - source cutouts are shared across splits, so this is a compositional split,
    not a novel-instance split.
  - ToolRGS must use a v2-aware GraspToolDataset that expands the queries list.

Recommended ToolRGS configuration:
  word_len: 32
"""
    (out_dir / "README.txt").write_text(text, encoding="utf-8")


def generate_dataset(config: GeneratorConfig) -> Path:
    src_dir = Path(config.src_dir).expanduser().resolve()
    background_dir = Path(config.background_dir).expanduser().resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if not background_dir.is_dir():
        raise FileNotFoundError(f"Background directory not found: {background_dir}")

    source_objects, source_warnings = load_source_objects(src_dir)
    objects_by_category: Dict[str, List[SourceObject]] = defaultdict(list)
    for source in source_objects:
        objects_by_category[source.category_key].append(source)
    missing_categories = sorted(set(CANONICAL_CATEGORY_NAMES) - set(objects_by_category))
    if missing_categories:
        raise ValueError(f"Missing canonical categories: {missing_categories}")

    background_paths = list_images(background_dir)
    background_splits = split_backgrounds(background_paths, config.seed)
    out_dir = prepare_output_directory(config, src_dir, background_dir)
    write_readme(out_dir, config)

    print(f"[source] valid objects: {len(source_objects)}")
    print(f"[source] categories: {len(objects_by_category)}")
    print(f"[source] backgrounds: {len(background_paths)}")
    for warning in source_warnings:
        print(f"[source-warning] {warning}")
    for split, values in background_splits.items():
        print(f"[backgrounds] {split}: {len(values)}")

    rng = random.Random(config.seed)
    category_sampler = BalancedCategorySampler(sorted(objects_by_category), rng)
    transform_sampler = BalancedTransformSampler(
        objects_by_category, config.scales, config.angle_bins, rng
    )
    prepared_cache: Dict[str, PreparedObject] = {}
    scene_counts = {
        "train": config.train_scenes,
        "val": config.val_scenes,
        "test": config.test_scenes,
    }
    stats: Dict[str, Any] = {
        "scenes": Counter(),
        "objects": Counter(),
        "queries": Counter(),
        "query_types": Counter(),
        "difficulty": Counter(),
        "category_placements": Counter(),
    }
    start_time = time.time()
    preview_written = 0

    for split in ("train", "val", "test"):
        index_path = out_dir / split / "index.jsonl"
        with index_path.open("w", encoding="utf-8") as index_file:
            for scene_number in range(scene_counts[split]):
                result = build_scene(
                    split,
                    scene_number,
                    background_splits[split],
                    objects_by_category,
                    prepared_cache,
                    category_sampler,
                    transform_sampler,
                    rng,
                    config,
                )
                if result is None:
                    raise RuntimeError(
                        f"Could not build {split} scene {scene_number} after "
                        f"{config.scene_attempts} attempts. Reduce objects/scales or relation constraints."
                    )
                image, annotation = result
                scene_id = annotation["scene_id"]
                image_path = out_dir / split / annotation["image_filename"]
                json_path = out_dir / split / f"{scene_id}.json"
                save_image(image_path, image, config)
                json_path.write_text(
                    json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                for query_index, query in enumerate(annotation["queries"]):
                    index_file.write(json.dumps({
                        "image": annotation["image_filename"],
                        "annotation": json_path.name,
                        "query_index": query_index,
                        "query_id": query["query_id"],
                        "target_idx": query["target_idx"],
                    }, ensure_ascii=False) + "\n")

                if preview_written < config.preview_count:
                    preview = draw_preview(image, annotation)
                    preview_path = out_dir / "_preview" / f"{split}_{scene_id}.jpg"
                    if not cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                        raise IOError(f"Failed to write preview: {preview_path}")
                    query_text = "\n".join(
                        f"[{q['target_idx']}] {q['type']}: {q['text']}"
                        for q in annotation["queries"]
                    )
                    preview_path.with_suffix(".txt").write_text(query_text, encoding="utf-8")
                    preview_written += 1

                stats["scenes"][split] += 1
                stats["objects"][split] += len(annotation["objects"])
                stats["queries"][split] += len(annotation["queries"])
                for obj in annotation["objects"]:
                    stats["category_placements"][obj["category"]] += 1
                for query in annotation["queries"]:
                    stats["query_types"][query["type"]] += 1
                    stats["difficulty"][str(query["difficulty"])] += 1

                done = scene_number + 1
                if done == scene_counts[split] or done % 100 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{split}] {done}/{scene_counts[split]} scenes | "
                        f"elapsed {elapsed / 60.0:.1f} min"
                    )

    serializable_stats = {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in stats.items()
    }
    metadata = {
        "generator": "tools/dataset_converters/grasp_tools/augment.py",
        "schema_version": "2.0",
        "config": asdict(config),
        "canonical_categories": list(CANONICAL_CATEGORY_NAMES.values()),
        "source_category_counts": {
            CANONICAL_CATEGORY_NAMES[key]: len(value)
            for key, value in sorted(objects_by_category.items())
        },
        "source_warnings": source_warnings,
        "background_splits": {
            split: [str(path) for path in paths]
            for split, paths in background_splits.items()
        },
        "stats": serializable_stats,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] dataset written to: {out_dir}")
    print(json.dumps(serializable_stats, ensure_ascii=False, indent=2))
    return out_dir


def main() -> int:
    try:
        config = build_config(parse_args())
        generate_dataset(config)
        return 0
    except KeyboardInterrupt:
        print("[cancelled] interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
