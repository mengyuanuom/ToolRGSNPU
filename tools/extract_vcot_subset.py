"""Extract only the Grasp-Anything files referenced by VCoT split CSVs.

The official image archive is distributed as raw consecutive parts. This tool
exposes those parts as one seekable stream, so no temporary 65 GB image.zip is
required. Extraction is resumable: an existing file with the expected size is
skipped, while new files are written atomically through a .partial file.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable, Sequence
from zipfile import BadZipFile, ZipFile, ZipInfo


SPLIT_FILES = {
    "train": "train.csv",
    "seen": "test_seen.csv",
    "unseen": "test_unseen.csv",
}


class ConcatenatedParts(io.RawIOBase):
    """Read raw split files as one seekable binary stream."""

    def __init__(self, paths: Sequence[Path]):
        super().__init__()
        if not paths:
            raise ValueError("At least one image archive part is required")
        self.paths = [Path(path).expanduser().resolve() for path in paths]
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"Image archive part not found: {path}")
        self._streams = [path.open("rb") for path in self.paths]
        self._sizes = [path.stat().st_size for path in self.paths]
        self._starts = []
        total = 0
        for size in self._sizes:
            self._starts.append(total)
            total += size
        self._length = total
        self._position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_CUR:
            offset += self._position
        elif whence == os.SEEK_END:
            offset += self._length
        elif whence != os.SEEK_SET:
            raise ValueError(f"Unsupported whence: {whence}")
        if offset < 0:
            raise ValueError("Cannot seek before the concatenated stream")
        self._position = int(offset)
        return self._position

    def read(self, size=-1):
        if self.closed:
            raise ValueError("I/O operation on closed stream")
        remaining = self._length - self._position
        if size is None or size < 0:
            size = remaining
        size = min(int(size), remaining)
        if size <= 0:
            return b""

        chunks = []
        while size > 0:
            part_index = self._part_index(self._position)
            local_offset = self._position - self._starts[part_index]
            available = self._sizes[part_index] - local_offset
            take = min(size, available)
            stream = self._streams[part_index]
            stream.seek(local_offset)
            chunk = stream.read(take)
            if not chunk:
                break
            chunks.append(chunk)
            consumed = len(chunk)
            self._position += consumed
            size -= consumed
        return b"".join(chunks)

    def _part_index(self, position):
        for index in range(len(self._starts) - 1, -1, -1):
            if position >= self._starts[index]:
                return index
        return 0

    def close(self):
        if not self.closed:
            for stream in self._streams:
                stream.close()
        super().close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a compact Grasp-Anything subset from VCoT CSVs"
    )
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_FILES),
        default=tuple(SPLIT_FILES),
    )
    parser.add_argument(
        "--image-parts",
        type=Path,
        nargs="+",
        required=True,
        help="Raw image ZIP parts in order, normally image_part_aa image_part_ab",
    )
    parser.add_argument("--positive-zip", type=Path, required=True)
    parser.add_argument("--mask-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all members and report size without extracting",
    )
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def load_split_ids(split_root: Path, splits: Iterable[str]):
    split_root = split_root.expanduser().resolve()
    grasp_ids = set()
    scene_ids = set()
    selected_files = []
    split_counts = {}
    for split in splits:
        filename = SPLIT_FILES[split]
        path = split_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"VCoT split CSV not found: {path}")
        count = 0
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row_number, row in enumerate(csv.reader(stream), start=1):
                if not row:
                    continue
                if len(row) < 3:
                    raise ValueError(
                        f"Malformed row {row_number} in {path}: {row!r}"
                    )
                grasp_id = row[0].strip()
                if not grasp_id or "_" not in grasp_id:
                    raise ValueError(
                        f"Invalid grasp id at row {row_number} in {path}: {grasp_id!r}"
                    )
                grasp_ids.add(grasp_id)
                scene_ids.add(grasp_id.rsplit("_", 1)[0])
                count += 1
        selected_files.append(path)
        split_counts[split] = count
    return grasp_ids, scene_ids, selected_files, split_counts


def build_plan(
    archive: ZipFile,
    identifiers: Iterable[str],
    prefix: str,
    suffix: str,
):
    plan = []
    missing = []
    for identifier in identifiers:
        member = f"{prefix}/{identifier}{suffix}"
        try:
            info = archive.getinfo(member)
        except KeyError:
            missing.append(member)
        else:
            plan.append(info)
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} requested members are missing from the archive. "
            f"First entries:\n{preview}"
        )
    plan.sort(key=lambda info: info.header_offset)
    return plan


def plan_summary(name: str, plan: Sequence[ZipInfo]):
    return {
        "name": name,
        "files": len(plan),
        "uncompressed_bytes": sum(info.file_size for info in plan),
        "compressed_bytes": sum(info.compress_size for info in plan),
    }


def human_bytes(value):
    value = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0


def extract_plan(
    archive: ZipFile,
    plan: Sequence[ZipInfo],
    output_root: Path,
    progress_every: int,
):
    extracted = 0
    skipped = 0
    started = time.time()
    for index, info in enumerate(plan, start=1):
        destination = output_root / Path(info.filename)
        if destination.is_file() and destination.stat().st_size == info.file_size:
            skipped += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            if partial.exists():
                partial.unlink()
            try:
                with archive.open(info, "r") as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                if partial.stat().st_size != info.file_size:
                    raise IOError(
                        f"Size mismatch for {info.filename}: "
                        f"{partial.stat().st_size} != {info.file_size}"
                    )
                os.replace(partial, destination)
                extracted += 1
            except Exception:
                if partial.exists():
                    partial.unlink()
                raise

        if progress_every > 0 and (
            index % progress_every == 0 or index == len(plan)
        ):
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[{index:>7}/{len(plan)}] extracted={extracted} "
                f"skipped={skipped} rate={index / elapsed:.1f} files/s",
                flush=True,
            )
    return {"extracted": extracted, "skipped": skipped}


def copy_splits(split_files: Sequence[Path], output_root: Path):
    destination_root = output_root / "split" / "vcot"
    destination_root.mkdir(parents=True, exist_ok=True)
    for source in split_files:
        shutil.copy2(source, destination_root / source.name)


def main():
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    positive_zip = args.positive_zip.expanduser().resolve()
    mask_zip = args.mask_zip.expanduser().resolve()
    for path in (positive_zip, mask_zip):
        if not path.is_file():
            raise FileNotFoundError(f"Source archive not found: {path}")

    grasp_ids, scene_ids, split_files, split_counts = load_split_ids(
        args.split_root, args.splits
    )
    print(
        f"Selected {len(grasp_ids)} grasp samples across "
        f"{len(scene_ids)} unique scenes"
    )

    image_stream = ConcatenatedParts(args.image_parts)
    try:
        with ZipFile(image_stream) as image_archive, ZipFile(
            positive_zip
        ) as positive_archive, ZipFile(mask_zip) as mask_archive:
            image_plan = build_plan(image_archive, scene_ids, "image", ".jpg")
            positive_plan = build_plan(
                positive_archive,
                grasp_ids,
                "grasp_label_positive",
                ".pt",
            )
            mask_plan = build_plan(mask_archive, grasp_ids, "mask", ".npy")
            summaries = [
                plan_summary("image", image_plan),
                plan_summary("grasp_label_positive", positive_plan),
                plan_summary("mask", mask_plan),
            ]
            total_bytes = sum(item["uncompressed_bytes"] for item in summaries)
            print(json.dumps(summaries, indent=2))
            print(f"Expected extracted payload: {human_bytes(total_bytes)}")

            if args.dry_run:
                return 0

            output_root.mkdir(parents=True, exist_ok=True)
            results = {}
            for name, archive, plan in (
                ("image", image_archive, image_plan),
                ("grasp_label_positive", positive_archive, positive_plan),
                ("mask", mask_archive, mask_plan),
            ):
                print(f"Extracting {name}: {len(plan)} files", flush=True)
                results[name] = extract_plan(
                    archive,
                    plan,
                    output_root,
                    max(0, args.progress_every),
                )
            copy_splits(split_files, output_root)
            manifest = {
                "format": "vcot-grasp-subset-v1",
                "splits": list(args.splits),
                "split_counts": split_counts,
                "unique_grasp_ids": len(grasp_ids),
                "unique_scene_ids": len(scene_ids),
                "payload": summaries,
                "results": results,
                "sources": {
                    "image_parts": [
                        os.fspath(Path(path).expanduser().resolve())
                        for path in args.image_parts
                    ],
                    "positive_zip": os.fspath(positive_zip),
                    "mask_zip": os.fspath(mask_zip),
                },
            }
            with (output_root / "vcot_subset_manifest.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
    except BadZipFile as exc:
        raise BadZipFile(f"Source archive validation failed: {exc}") from exc
    finally:
        image_stream.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
