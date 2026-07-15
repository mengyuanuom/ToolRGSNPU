"""Inspect one VCoT CSV row and its lazy-loaded Grasp-Anything files."""

import argparse
import csv
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--row", type=int, default=0, help="Zero-based CSV row")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.csv.open("r", encoding="utf-8", newline="") as stream:
        try:
            row = next(islice(csv.reader(stream), args.row, args.row + 1))
        except StopIteration as error:
            raise IndexError(f"CSV row {args.row} does not exist: {args.csv}") from error
    grasp_id, object_name, *description = row
    description = ",".join(description)
    scene_id = grasp_id.rsplit("_", 1)[0]
    image_path = args.dataset_root / "image" / f"{scene_id}.jpg"
    grasp_path = args.dataset_root / "positive_grasp" / f"{grasp_id}.pt"
    mask_path = args.dataset_root / "mask" / f"{grasp_id}.npy"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    try:
        grasps = torch.load(grasp_path, map_location="cpu", weights_only=True)
    except TypeError:
        grasps = torch.load(grasp_path, map_location="cpu")
    mask = np.load(mask_path, allow_pickle=False)

    print(f"CSV:         {args.csv.resolve()}")
    print(f"CSV row:     {args.row}")
    print(f"grasp_id:    {grasp_id}")
    print(f"object:      {object_name}")
    print(f"description: {description}")
    print(f"image:       {image_path.resolve()}")
    print(f"image shape: {None if image is None else image.shape}")
    print(f"grasp:       {grasp_path.resolve()}")
    print(f"grasp value: type={type(grasps).__name__}, shape={getattr(grasps, 'shape', None)}")
    print(f"first grasps:\n{grasps[:5]}")
    print(f"mask:        {mask_path.resolve()}")
    print(
        f"mask value:  shape={mask.shape}, dtype={mask.dtype}, "
        f"min={mask.min()}, max={mask.max()}, nonzero={np.count_nonzero(mask)}"
    )


if __name__ == "__main__":
    main()
