#!/usr/bin/env python3
"""Render a small, deterministic test-set demo for a trained grasp model."""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.inference import ToolRGSInference
from toolrgs.datasets import build_dataset
import utils.config as config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select distinct test scenes and draw predicted segmentation masks "
            "and grasp rectangles on their original images."
        )
    )
    parser.add_argument(
        "--config",
        default="config/grasp_tools/crog.yaml",
        help="Experiment config, relative to the repository root by default.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help=(
            "Checkpoint path. When omitted, the newest best-J@1 checkpoint for "
            "the configured experiment is selected automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="demo_outputs/crog_test",
        help="Directory for annotated images and manifest.json.",
    )
    parser.add_argument(
        "--split",
        default="",
        help="Dataset split; defaults to TEST.test_split from the config.",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-grasps", type=int, default=5)
    parser.add_argument("--npu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--mask-threshold", type=float, default=0.35)
    parser.add_argument("--quality-threshold", type=float, default=0.4)
    return parser.parse_args()


def resolve_repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_checkpoint(checkpoint_arg, cfg):
    if checkpoint_arg:
        checkpoint = resolve_repo_path(checkpoint_arg)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    output_root = resolve_repo_path(cfg.output_folder)
    experiment_prefix = str(cfg.exp_name)
    experiment_dirs = []
    exact = output_root / experiment_prefix
    if exact.is_dir():
        experiment_dirs.append(exact)
    if output_root.is_dir():
        experiment_dirs.extend(
            directory
            for directory in output_root.iterdir()
            if directory.is_dir()
            and directory.name.startswith(f"{experiment_prefix}_")
            and directory not in experiment_dirs
        )

    preferred_patterns = (
        "best_j1_epoch_*.pth",
        "best_jindex_model.pth",
        "best_iou_epoch_*.pth",
        "best_iou_model.pth",
        "last.pth",
    )
    for pattern in preferred_patterns:
        candidates = [
            checkpoint
            for directory in experiment_dirs
            for checkpoint in directory.glob(pattern)
            if checkpoint.is_file()
        ]
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(
        "No CROG checkpoint was found automatically under "
        f"{output_root}. Pass one explicitly with --checkpoint."
    )


def select_distinct_scene_indices(dataset, count, seed):
    if count <= 0:
        raise ValueError("--num-samples must be positive")
    rng = random.Random(seed)
    samples = getattr(dataset, "samples", None)
    if not samples:
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        return indices[: min(count, len(indices))]

    by_image = {}
    for index, record in enumerate(samples):
        image_path = record[0] if isinstance(record, (tuple, list)) else index
        by_image.setdefault(str(image_path), []).append(index)
    image_paths = list(by_image)
    rng.shuffle(image_paths)
    selected = [
        rng.choice(by_image[image_path])
        for image_path in image_paths[: min(count, len(image_paths))]
    ]
    return selected


def safe_stem(value):
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return normalized[:80] or "sample"


def wrap_for_image(text, width, font_scale=0.55, thickness=1):
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        text_width = cv2.getTextSize(
            candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0][0]
        if text_width <= max(80, width - 20):
            line = candidate
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return lines


def add_demo_header(image, ordinal, total, sample):
    scene = sample.get("scene_id", "unknown")
    query_type = sample.get("query_type", "unknown")
    target = sample.get("target", "unknown")
    meta = (
        f"{ordinal:02d}/{total:02d}  scene={scene}  "
        f"target={target}  query={query_type}"
    )
    prompt_lines = wrap_for_image(
        f"Prompt: {sample.get('sentence', '')}", image.shape[1]
    )
    legend = "green=predicted mask  yellow=predicted grasp  red=center"
    lines = [meta, *prompt_lines, legend]
    line_height = 24
    header = np.zeros((line_height * len(lines) + 10, image.shape[1], 3), np.uint8)
    for line_index, line in enumerate(lines):
        cv2.putText(
            header,
            line,
            (8, 21 + line_index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.vconcat([header, image])


def main():
    args = parse_args()
    if args.num_grasps <= 0:
        raise ValueError("--num-grasps must be positive")

    os.chdir(REPO_ROOT)
    config_path = resolve_repo_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")
    cfg = config.load_cfg_from_cfg_file(str(config_path))
    cfg.root_path = str(resolve_repo_path(cfg.root_path))
    split = args.split or cfg.test_split
    checkpoint = resolve_checkpoint(args.checkpoint, cfg)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    needs_offset = str(cfg.architecture).lower() in {"crogoff", "drogoff"}
    dataset = build_dataset(cfg, split, with_offset=needs_offset)
    selected_indices = select_distinct_scene_indices(
        dataset, args.num_samples, args.seed
    )
    if len(selected_indices) < args.num_samples:
        print(
            f"Requested {args.num_samples} samples, but split {split!r} only "
            f"contains {len(selected_indices)} distinct images."
        )

    predictor = ToolRGSInference(
        {
            "_repo_root": str(REPO_ROOT),
            "model": {
                "config": str(config_path),
                "checkpoint": str(checkpoint),
                "device": f"npu:{args.npu}",
                "mask_threshold": args.mask_threshold,
                "quality_threshold": args.quality_threshold,
                "num_grasps": args.num_grasps,
                "gate_quality_by_mask": True,
                "scale_grasp_to_source": True,
                "postprocessor": {
                    "type": "dense_grasp",
                    "min_distance": 2,
                    "width_factor": 100.0,
                    "grasp_height": 20.0,
                },
                "overrides": {},
            },
        }
    )

    manifest = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "split": split,
        "seed": args.seed,
        "samples": [],
    }
    total = len(selected_indices)
    for ordinal, dataset_index in enumerate(selected_indices, start=1):
        sample = dataset[dataset_index]
        image_path = Path(sample["img_path"])
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Failed to read test image: {image_path}")
        prediction = predictor.predict(frame, sample["sentence"])
        rendered = add_demo_header(
            prediction.annotated_bgr, ordinal, total, sample
        )
        filename = (
            f"{ordinal:02d}_{safe_stem(sample.get('scene_id', dataset_index))}_"
            f"{safe_stem(sample.get('sent_id', 'query'))}.jpg"
        )
        destination = output_dir / filename
        if not cv2.imwrite(str(destination), rendered):
            raise OSError(f"Failed to save visualization: {destination}")
        manifest["samples"].append(
            {
                "dataset_index": dataset_index,
                "image": str(image_path),
                "output": str(destination),
                "scene_id": sample.get("scene_id"),
                "sent_id": sample.get("sent_id"),
                "target": sample.get("target"),
                "query_type": sample.get("query_type"),
                "prompt": prediction.prompt,
                "grasps": prediction.grasps,
                "scores": prediction.scores,
            }
        )
        print(f"[{ordinal:02d}/{total:02d}] saved {destination}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Checkpoint: {checkpoint}")
    print(f"Demo output: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
