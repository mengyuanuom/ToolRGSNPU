#!/usr/bin/env python3
"""Download the official backbone weights expected by ToolRGSNPU configs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Optional
import urllib.request


@dataclass(frozen=True)
class Artifact:
    filename: str
    url: str
    sha256: Optional[str] = None
    sha256_prefix: Optional[str] = None


ARTIFACTS = {
    "clip-rn50": Artifact(
        "RN50.pt",
        "https://openaipublic.azureedge.net/clip/models/"
        "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/"
        "RN50.pt",
        sha256="afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
    ),
    "clip-rn101": Artifact(
        "RN101.pt",
        "https://openaipublic.azureedge.net/clip/models/"
        "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/"
        "RN101.pt",
        sha256="8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599",
    ),
    "clip-vit-b16": Artifact(
        "ViT-B-16.pt",
        "https://openaipublic.azureedge.net/clip/models/"
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/"
        "ViT-B-16.pt",
        sha256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
    ),
    "dinov2-vitb14-reg4": Artifact(
        "dinov2_vitb14_reg4_pretrain.pth",
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
        "dinov2_vitb14_reg4_pretrain.pth",
    ),
    "mambavision-t": Artifact(
        "mambavision_tiny_1k.pth.tar",
        "https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/"
        "mambavision_tiny_1k.pth.tar",
        sha256="952a3e486f94bbe863c753a7ecabe282b2e3b8adbb0d98057e047e4f554c2a9b",
    ),
    "resnet18": Artifact(
        "resnet18-f37072fd.pth",
        "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        sha256_prefix="f37072fd",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_matches(path: Path, artifact: Artifact) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if not artifact.sha256 and not artifact.sha256_prefix:
        return True
    digest = file_sha256(path)
    if artifact.sha256:
        return digest == artifact.sha256
    return digest.startswith(artifact.sha256_prefix or "")


def download(name: str, output_dir: Path, force: bool = False) -> Path:
    artifact = ARTIFACTS[name]
    target = output_dir / artifact.filename
    if target.exists() and not force:
        if checksum_matches(target, artifact):
            print(f"[skip] {name}: {target}")
            return target
        raise RuntimeError(
            f"Existing file failed validation: {target}. "
            "Remove it or rerun with --force."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    request = urllib.request.Request(
        artifact.url, headers={"User-Agent": "ToolRGSNPU-weight-downloader/1.0"}
    )
    print(f"[download] {name}\n  from: {artifact.url}\n  to:   {target}")
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as stream:
            total = int(response.headers.get("Content-Length", 0))
            received = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                received += len(block)
                if total:
                    print(
                        f"\r  {received / 1024**2:.1f}/{total / 1024**2:.1f} MiB",
                        end="",
                        flush=True,
                    )
            if total:
                print()
        if not checksum_matches(temporary, artifact):
            raise RuntimeError(f"Checksum validation failed for {name}: {temporary}")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[ok] {target}")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifacts",
        nargs="*",
        choices=sorted(ARTIFACTS),
        help="Weights to download. With no names, only the manifest is printed.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Download every listed backbone weight."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pretrain"), help="Destination directory."
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing invalid or old file."
    )
    return parser


def print_manifest() -> None:
    print("Available pretrained artifacts:")
    for name, artifact in ARTIFACTS.items():
        print(f"  {name:22} -> pretrain/{artifact.filename}")


def main() -> int:
    args = build_parser().parse_args()
    if args.all and args.artifacts:
        raise SystemExit("Use either --all or explicit artifact names, not both.")
    names = list(ARTIFACTS) if args.all else args.artifacts
    if not names:
        print_manifest()
        print("\nExample: python tools/download_pretrained.py clip-rn50")
        return 0
    try:
        for name in names:
            download(name, args.output_dir, force=args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
