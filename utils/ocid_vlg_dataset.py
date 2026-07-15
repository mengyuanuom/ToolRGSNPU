"""OCID-VLG dataset adapter for the unified ToolRGS data contract."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.dataset import GraspTransforms, make_dense_offset_with_radius_np, tokenize
from utils.OCID_sub_class_dict import subnames


_SPLIT_FILES = {
    "train": "train_expressions.json",
    "val": "val_expressions.json",
    "validation": "val_expressions.json",
    "test": "test_expressions.json",
}


def resolve_ocid_vlg_split(split):
    """Return the official expression filename for an OCID-VLG split alias."""
    key = str(split).strip().lower().replace("-", "_")
    if key not in _SPLIT_FILES:
        choices = ", ".join(sorted(_SPLIT_FILES))
        raise ValueError(f"Unknown OCID-VLG split {split!r}; choose one of: {choices}")
    return _SPLIT_FILES[key]


def parse_ocid_image_filename(value):
    """Parse the official ``sequence_path,image_name`` metadata field."""
    parts = [part.strip() for part in str(value).split(",", 1)]
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "OCID-VLG image_filename must be 'sequence_path,image_name', "
            f"got {value!r}"
        )
    return parts[0], parts[1]


class OCIDVLGDataset(Dataset):
    """Load referring expressions, instance masks, depth, and grasp rectangles."""

    def __init__(
        self,
        root_dir,
        input_size=416,
        split="train",
        word_length=17,
        version="multiple",
        with_depth=True,
        with_offset=False,
        offset_radius=20.0,
        offset_sigma=None,
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.input_size = (int(input_size), int(input_size))
        self.word_length = int(word_length)
        self.version = str(version)
        self.with_depth = bool(with_depth)
        self.with_offset = bool(with_offset)
        self.offset_radius = float(offset_radius)
        self.offset_sigma = offset_sigma
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
        self.grasp_transform = GraspTransforms(
            width_factor=100, width=self.input_size[1], height=self.input_size[0]
        )

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"OCID-VLG root not found: {self.root_dir}")
        self.expression_path = (
            self.root_dir
            / "refer"
            / self.version
            / resolve_ocid_vlg_split(split)
        )
        if not self.expression_path.is_file():
            raise FileNotFoundError(
                f"OCID-VLG expression split not found: {self.expression_path}. "
                "Check DATA.root_path, DATA.version, and DATA train/val split names."
            )

        with self.expression_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        self.samples = metadata.get("data") if isinstance(metadata, dict) else None
        if not isinstance(self.samples, list) or not self.samples:
            raise ValueError(
                f"OCID-VLG split must contain a non-empty 'data' list: {self.expression_path}"
            )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _require(path, kind, sample_id):
        if not path.is_file():
            raise FileNotFoundError(f"Missing OCID-VLG {kind} for {sample_id}: {path}")
        return path

    @staticmethod
    def _transform_matrix(image_size, input_size):
        ori_h, ori_w = image_size
        inp_h, inp_w = input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2.0, (inp_h - new_h) / 2.0
        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], dtype=np.float32)
        dst = np.array(
            [[bias_x, bias_y], [new_w + bias_x, bias_y], [bias_x, new_h + bias_y]],
            dtype=np.float32,
        )
        return cv2.getAffineTransform(src, dst), cv2.getAffineTransform(dst, src)

    @staticmethod
    def _apply_affine(points, matrix):
        if not len(points):
            return np.zeros_like(points, dtype=np.float32)
        ones = np.ones((*points.shape[:-1], 1), dtype=np.float32)
        homogeneous = np.concatenate([points.astype(np.float32), ones], axis=-1)
        return np.einsum("ij,...j->...i", matrix, homogeneous).astype(np.float32)

    @staticmethod
    def _transform_bbox(box_xywh, matrix, input_size):
        x, y, width, height = [float(value) for value in box_xywh]
        corners = np.array([[x, y], [x + width, y + height]], dtype=np.float32)
        corners = OCIDVLGDataset._apply_affine(corners, matrix)
        inp_h, inp_w = input_size
        x1, y1 = np.maximum(corners[0], [0.0, 0.0])
        x2, y2 = np.minimum(corners[1], [float(inp_w), float(inp_h)])
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _read_image(path, flag, kind):
        value = cv2.imread(os.fspath(path), flag)
        if value is None:
            raise ValueError(f"OpenCV could not decode OCID-VLG {kind}: {path}")
        return value

    def __getitem__(self, index):
        item = self.samples[index]
        required = ("image_filename", "box", "grasps", "answer", "target", "question")
        missing = [key for key in required if key not in item]
        if missing:
            raise KeyError(
                f"OCID-VLG sample {index} is missing fields {missing}: {self.expression_path}"
            )

        sequence_path, image_name = parse_ocid_image_filename(item["image_filename"])
        sent_id = item.get("question_index", index)
        sample_id = f"{item['image_filename']}#{sent_id}"
        sequence_root = self.root_dir / sequence_path
        image_path = self._require(sequence_root / "rgb" / image_name, "image", sample_id)
        mask_path = self._require(
            sequence_root / "seg_mask_instances_combi" / image_name,
            "instance mask",
            sample_id,
        )

        image = self._read_image(image_path, cv2.IMREAD_COLOR, "image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = image.shape[:2]
        instance_mask = self._read_image(mask_path, cv2.IMREAD_UNCHANGED, "instance mask")
        if instance_mask.ndim != 2 or instance_mask.shape != (ori_h, ori_w):
            raise ValueError(
                f"OCID-VLG mask/image mismatch for {sample_id}: "
                f"mask={instance_mask.shape}, image={(ori_h, ori_w)}"
            )
        object_id = int(item["answer"])
        target_mask = (instance_mask == object_id).astype(np.uint8)
        if not target_mask.any():
            raise ValueError(
                f"OCID-VLG object id {object_id} has no pixels in {mask_path} ({sample_id})"
            )

        grasps = np.asarray(item["grasps"], dtype=np.float32)
        if grasps.ndim == 2 and grasps.shape == (4, 2):
            grasps = grasps[None, ...]
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 2) or not len(grasps):
            raise ValueError(
                f"OCID-VLG grasps must have shape [N,4,2], got {grasps.shape} ({sample_id})"
            )
        valid = np.isfinite(grasps).all(axis=(1, 2))
        grasps = grasps[valid]
        if not len(grasps):
            raise ValueError(f"OCID-VLG sample has no finite grasps: {sample_id}")

        target_name = str(item["target"])
        if target_name not in subnames:
            raise KeyError(f"Unknown OCID-VLG target instance {target_name!r}: {sample_id}")
        target_idx = int(subnames[target_name])
        original_targets = self.grasp_transform(grasps, target=target_idx)

        matrix, inverse = self._transform_matrix((ori_h, ori_w), self.input_size)
        input_grasps = self._apply_affine(grasps, matrix)
        input_targets = self.grasp_transform(input_grasps, target=target_idx)
        raw_masks = self.grasp_transform.generate_masks(input_targets)
        angle = raw_masks["ang"].astype(np.float32) * np.pi / 180.0
        grasp_masks = {
            "qua": torch.from_numpy(raw_masks["qua"].astype(np.float32) / 255.0),
            "sin": torch.from_numpy(np.sin(2.0 * angle)).float(),
            "cos": torch.from_numpy(np.cos(2.0 * angle)).float(),
            "wid": torch.from_numpy(raw_masks["wid"].astype(np.float32) / 255.0),
        }
        if self.with_offset:
            offsets, offset_weights = make_dense_offset_with_radius_np(
                centers_xy=input_targets[:, :2],
                img_size_hw=self.input_size,
                r_pix=self.offset_radius,
                use_gaussian=True,
                sigma=self.offset_sigma,
            )
            grasp_masks["off"] = torch.from_numpy(offsets).float()
            grasp_masks["off_w"] = torch.from_numpy(offset_weights).float()

        image = cv2.warpAffine(
            image,
            matrix,
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image.div_(255.0).sub_(self.mean).div_(self.std)
        target_mask = cv2.warpAffine(
            target_mask,
            matrix,
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if self.with_depth:
            depth_path = self._require(
                sequence_root / "depth" / image_name, "depth", sample_id
            )
            depth = self._read_image(depth_path, cv2.IMREAD_UNCHANGED, "depth").astype(
                np.float32
            )
            if depth.shape != (ori_h, ori_w):
                raise ValueError(
                    f"OCID-VLG depth/image mismatch for {sample_id}: "
                    f"depth={depth.shape}, image={(ori_h, ori_w)}"
                )
            depth = depth / 1000.0
            depth = cv2.warpAffine(
                depth,
                matrix,
                (self.input_size[1], self.input_size[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            depth_path = None
            depth = np.zeros(self.input_size, dtype=np.float32)

        sentence = str(item["question"])
        word_vec = tokenize(sentence, self.word_length, True).squeeze(0).long()
        if word_vec.numel() < self.word_length:
            word_vec = torch.cat(
                [word_vec, torch.zeros(self.word_length - word_vec.numel(), dtype=torch.long)]
            )
        else:
            word_vec = word_vec[: self.word_length]

        original_box = np.asarray(item["box"], dtype=np.float32)
        if original_box.shape != (4,):
            raise ValueError(f"OCID-VLG box must be [x,y,w,h], got {original_box}: {sample_id}")

        return {
            "img": image,
            "depth": torch.from_numpy(depth).unsqueeze(0).float(),
            "mask": torch.from_numpy(target_mask).float(),
            "grasp_masks": grasp_masks,
            "word_vec": word_vec,
            "grasps": original_targets,
            "target": target_name,
            "sentence": sentence,
            "bbox": self._transform_bbox(original_box, matrix, self.input_size),
            "bbox_original": np.array(
                [
                    original_box[0],
                    original_box[1],
                    original_box[0] + original_box[2],
                    original_box[1] + original_box[3],
                ],
                dtype=np.float32,
            ),
            "target_idx": target_idx,
            "object_id": object_id,
            "program": item.get("program"),
            "sent_id": sent_id,
            "scene_id": str(item["image_filename"]),
            "expression_path": os.fspath(self.expression_path),
            "inverse": inverse,
            "ori_size": np.array([ori_h, ori_w]),
            "img_path": os.fspath(image_path),
            "depth_path": None if depth_path is None else os.fspath(depth_path),
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

        stacked = {
            "img": torch.stack([sample["img"] for sample in batch]),
            "depth": torch.stack([sample["depth"] for sample in batch]),
            "mask": torch.stack([sample["mask"] for sample in batch]),
            "grasp_masks": grasp_masks,
            "word_vec": torch.stack([sample["word_vec"] for sample in batch]),
        }
        list_keys = (
            "grasps", "target", "sentence", "bbox", "bbox_original",
            "target_idx", "object_id", "program", "sent_id", "scene_id",
            "expression_path", "inverse", "ori_size", "img_path", "depth_path",
            "mask_path",
        )
        stacked.update({key: [sample[key] for sample in batch] for key in list_keys})
        return stacked
