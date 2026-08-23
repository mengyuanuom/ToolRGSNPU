"""Create the minimal GraspNet_VLG tree required by ToolRGS RealVLG configs.

The compact tree keeps:

* all Kinect metadata needed to reproduce the configured deterministic train
  subset;
* RGB images and SAM2 object masks referenced by that train subset;
* frame 0000 metadata, RGB images, and masks for all three official
  evaluation splits (seen, similar, and novel).

Everything else in the public archive (the Realsense camera, duplicate
``metadata_filter`` files, unused training RGB/masks, and helper scripts) is
excluded. Grasp corners, descriptions, labels, and bounding boxes remain in
the retained JSON metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


TRAIN_SCENES = range(0, 100)
EVAL_SPLITS = {
    "seen": range(100, 130),
    "similar": range(130, 160),
    "novel": range(160, 190),
}


@dataclass(frozen=True)
class Sample:
    key: str
    metadata_path: str
    image_path: str
    mask_path: str


def stable_rank(sample_key: str, seed: int) -> int:
    payload = f"{int(seed)}:{sample_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def normalized_member_path(value: object, *, kind: str, sample_key: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"Sample {sample_key} has no {kind} path")
    member = str(value).lstrip("/\\").replace("\\", "/")
    parts = PurePosixPath(member).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            f"Unsafe {kind} archive path for sample {sample_key}: {value!r}"
        )
    return PurePosixPath(*parts).as_posix()


def metadata_member(camera: str, scene_id: int, frame: str) -> str:
    return f"metadata/{camera}/scene_{scene_id:04d}/{frame}.json"


def load_metadata_samples(
    archive: zipfile.ZipFile, member: str
) -> list[Sample]:
    with archive.open(member) as stream:
        objects = json.load(stream)
    if not isinstance(objects, list):
        raise ValueError(f"Metadata must contain a list: {member}")

    relative = member.split("/", 2)[2]
    samples: list[Sample] = []
    for object_index, item in enumerate(objects):
        if not isinstance(item, dict) or not item.get("grasps"):
            continue
        object_id = str(item.get("object_id", object_index))
        sample_key = f"{relative}#{object_id}"
        samples.append(
            Sample(
                key=sample_key,
                metadata_path=member,
                image_path=normalized_member_path(
                    item.get("image_path"), kind="image", sample_key=sample_key
                ),
                mask_path=normalized_member_path(
                    item.get("mask_path"), kind="mask", sample_key=sample_key
                ),
            )
        )
    return samples


def build_plan(
    archive: zipfile.ZipFile,
    *,
    camera: str,
    train_fraction: float,
    train_seed: int,
    eval_frame: str,
) -> dict:
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")

    archive_names = set(archive.namelist())
    metadata_paths: set[str] = set()
    train_samples: list[Sample] = []
    missing_metadata: list[str] = []

    print("Scanning Kinect training metadata...", flush=True)
    for scene_id in TRAIN_SCENES:
        prefix = f"metadata/{camera}/scene_{scene_id:04d}/"
        scene_members = sorted(
            name
            for name in archive_names
            if name.startswith(prefix) and name.endswith(".json")
        )
        if not scene_members:
            missing_metadata.append(prefix + "*.json")
            continue
        for member in scene_members:
            metadata_paths.add(member)
            train_samples.extend(load_metadata_samples(archive, member))
        print(
            f"  scenes 0000-{scene_id:04d}: "
            f"{len(metadata_paths):,} metadata files, "
            f"{len(train_samples):,} graspable objects",
            flush=True,
        )

    if missing_metadata:
        raise FileNotFoundError(
            "Archive is missing training metadata: "
            + ", ".join(missing_metadata[:10])
        )
    if not train_samples:
        raise ValueError("No graspable training objects were found")

    selected_count = max(1, int(round(len(train_samples) * train_fraction)))
    selected_train = sorted(
        train_samples, key=lambda sample: (stable_rank(sample.key, train_seed), sample.key)
    )[:selected_count]

    eval_samples: list[Sample] = []
    eval_counts: Counter[str] = Counter()
    print("Scanning official evaluation metadata...", flush=True)
    for split, scene_ids in EVAL_SPLITS.items():
        for scene_id in scene_ids:
            member = metadata_member(camera, scene_id, eval_frame)
            if member not in archive_names:
                missing_metadata.append(member)
                continue
            metadata_paths.add(member)
            found = load_metadata_samples(archive, member)
            eval_samples.extend(found)
            eval_counts[split] += len(found)

    if missing_metadata:
        raise FileNotFoundError(
            "Archive is missing required metadata: "
            + ", ".join(missing_metadata[:10])
        )

    selected_samples = selected_train + eval_samples
    image_paths = {sample.image_path for sample in selected_samples}
    mask_paths = {sample.mask_path for sample in selected_samples}
    payload_paths = image_paths | mask_paths
    required_paths = metadata_paths | payload_paths

    missing_payload = sorted(path for path in payload_paths if path not in archive_names)
    if missing_payload:
        raise FileNotFoundError(
            "Archive is missing referenced payload files: "
            + ", ".join(missing_payload[:10])
        )

    uncompressed = sum(archive.getinfo(path).file_size for path in required_paths)
    compressed = sum(archive.getinfo(path).compress_size for path in required_paths)
    return {
        "camera": camera,
        "train_fraction": train_fraction,
        "train_seed": train_seed,
        "eval_frame": eval_frame,
        "train_total_objects": len(train_samples),
        "train_selected_objects": len(selected_train),
        "train_selected_keys": [sample.key for sample in selected_train],
        "eval_objects": dict(eval_counts),
        "metadata_paths": sorted(metadata_paths),
        "image_paths": sorted(image_paths),
        "mask_paths": sorted(mask_paths),
        "required_paths": required_paths,
        "uncompressed_bytes": uncompressed,
        "compressed_bytes": compressed,
    }


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def print_plan(plan: dict, archive_size: int) -> None:
    print("\nCompact GraspNet_VLG plan")
    print(f"  camera:                 {plan['camera']}")
    print(f"  training objects:       {plan['train_total_objects']:,}")
    print(f"  selected train objects: {plan['train_selected_objects']:,}")
    for split in EVAL_SPLITS:
        print(
            f"  {split:8s} eval objects:  "
            f"{plan['eval_objects'].get(split, 0):,}"
        )
    print(f"  metadata files:         {len(plan['metadata_paths']):,}")
    print(f"  unique RGB images:      {len(plan['image_paths']):,}")
    print(f"  object masks:           {len(plan['mask_paths']):,}")
    print(f"  compact size:           {human_bytes(plan['uncompressed_bytes'])}")
    print(f"  source ZIP size:        {human_bytes(archive_size)}")
    saved = archive_size - plan["uncompressed_bytes"]
    print(f"  net space after ZIP deletion: {human_bytes(saved)}")


def safe_target(output_root: Path, member: str) -> Path:
    root = output_root.resolve()
    target = root.joinpath(*PurePosixPath(member).parts).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Archive path escapes output root: {member}")
    return target


def extract_plan(archive: zipfile.ZipFile, output_root: Path, plan: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    required = sorted(
        plan["required_paths"], key=lambda name: archive.getinfo(name).header_offset
    )
    total_bytes = int(plan["uncompressed_bytes"])
    completed_bytes = 0
    extracted_files = 0
    skipped_files = 0
    started = time.monotonic()

    print(f"\nExtracting {len(required):,} required files...", flush=True)
    for index, member in enumerate(required, start=1):
        info = archive.getinfo(member)
        target = safe_target(output_root, member)
        if target.is_file() and target.stat().st_size == info.file_size:
            completed_bytes += info.file_size
            skipped_files += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            if partial.exists():
                partial.unlink()
            try:
                with archive.open(info) as source, partial.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                if partial.stat().st_size != info.file_size:
                    raise IOError(
                        f"Extracted size mismatch for {member}: "
                        f"{partial.stat().st_size} != {info.file_size}"
                    )
                os.replace(partial, target)
            finally:
                if partial.exists():
                    partial.unlink()
            completed_bytes += info.file_size
            extracted_files += 1

        if index == 1 or index % 1000 == 0 or index == len(required):
            elapsed = max(time.monotonic() - started, 0.001)
            rate = completed_bytes / elapsed
            percent = 100.0 * completed_bytes / max(total_bytes, 1)
            print(
                f"  {index:,}/{len(required):,} files, {percent:5.1f}%, "
                f"{human_bytes(rate)}/s",
                flush=True,
            )

    info_path = output_root / "COMPACT_DATASET_INFO.json"
    info_payload = {
        "format": "ToolRGS compact GraspNet_VLG",
        "camera": plan["camera"],
        "train_fraction": plan["train_fraction"],
        "train_seed": plan["train_seed"],
        "train_total_objects": plan["train_total_objects"],
        "train_selected_objects": plan["train_selected_objects"],
        "train_selected_keys": plan["train_selected_keys"],
        "eval_frame": plan["eval_frame"],
        "eval_objects": plan["eval_objects"],
        "metadata_files": len(plan["metadata_paths"]),
        "rgb_images": len(plan["image_paths"]),
        "object_masks": len(plan["mask_paths"]),
        "uncompressed_bytes": plan["uncompressed_bytes"],
    }
    info_path.write_text(
        json.dumps(info_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Extraction complete: {extracted_files:,} written, "
        f"{skipped_files:,} already present.",
        flush=True,
    )


def verify_plan(archive: zipfile.ZipFile, output_root: Path, plan: dict) -> None:
    print("Verifying compact dataset file sizes...", flush=True)
    missing: list[str] = []
    mismatched: list[str] = []
    for member in plan["required_paths"]:
        target = safe_target(output_root, member)
        if not target.is_file():
            missing.append(member)
            continue
        expected = archive.getinfo(member).file_size
        actual = target.stat().st_size
        if actual != expected:
            mismatched.append(f"{member}: {actual} != {expected}")
    if missing or mismatched:
        details = missing[:10] + mismatched[:10]
        raise IOError("Compact dataset verification failed: " + "; ".join(details))
    print("Verification complete: every required file is present at the expected size.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera", default="kinect")
    parser.add_argument("--train-fraction", type=float, default=0.1)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-frame", default="0000")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the plan. Without this flag the command is read-only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_path = args.archive.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        plan = build_plan(
            archive,
            camera=args.camera,
            train_fraction=args.train_fraction,
            train_seed=args.train_seed,
            eval_frame=args.eval_frame.removesuffix(".json"),
        )
        print_plan(plan, archive_path.stat().st_size)
        if args.extract:
            extract_plan(archive, output_root, plan)
            verify_plan(archive, output_root, plan)
        else:
            print("\nDry run only; pass --extract to create the compact dataset.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, IOError, zipfile.BadZipFile) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
