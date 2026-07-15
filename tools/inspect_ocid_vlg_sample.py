"""Inspect one OCID-VLG expression and all files it resolves to."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SPLITS = {
    "train": "train_expressions.json",
    "val": "val_expressions.json",
    "test": "test_expressions.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--version", default="multiple")
    parser.add_argument("--split", choices=sorted(SPLITS), default="train")
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    expression_path = args.dataset_root / "refer" / args.version / SPLITS[args.split]
    with expression_path.open("r", encoding="utf-8") as stream:
        items = json.load(stream)["data"]
    item = items[args.index]
    sequence_path, image_name = [part.strip() for part in item["image_filename"].split(",", 1)]
    sequence_root = args.dataset_root / sequence_path
    image_path = sequence_root / "rgb" / image_name
    depth_path = sequence_root / "depth" / image_name
    mask_path = sequence_root / "seg_mask_instances_combi" / image_name

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    object_id = int(item["answer"])
    grasps = np.asarray(item["grasps"], dtype=np.float32)

    print(f"expressions:  {expression_path.resolve()}")
    print(f"sample index: {args.index}")
    print(f"sentence id:  {item.get('question_index')}")
    print(f"scene:        {item['image_filename']}")
    print(f"sentence:     {item['question']}")
    print(f"target:       {item['target']} (instance id {object_id})")
    print(f"bbox xywh:    {item['box']}")
    print(f"image:        {image_path.resolve()} shape={None if image is None else image.shape}")
    print(f"depth:        {depth_path.resolve()} shape={None if depth is None else depth.shape}")
    print(f"mask:         {mask_path.resolve()} shape={None if mask is None else mask.shape}")
    pixels = 0 if mask is None else int(np.count_nonzero(mask == object_id))
    print(f"object pixels: {pixels}")
    print(f"grasps:       shape={grasps.shape}")
    print(f"first grasp:\n{None if not len(grasps) else grasps[0]}")


if __name__ == "__main__":
    main()
