import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import yaml

from toolrgs.evaluation import (
    evaluate_realvlg_grasp,
    realvlg_angular_diff,
    realvlg_ciou,
    realvlg_e_measure,
    realvlg_f_measure,
    realvlg_giou,
    realvlg_mask_to_bbox,
    realvlg_rect_to_points8,
    realvlg_s_measure,
    resolve_evaluation_protocol,
)
from toolrgs.engine.realvlg_val_loop import RealVLGValLoop
from utils.realvlg_dataset import (
    REALVLG_EVAL_SCENES,
    RealVLGDataset,
    realvlg_points8_to_rectangles,
)


ROOT = Path(__file__).resolve().parents[1]


class RealVLGOfficialMetricTest(unittest.TestCase):
    def test_protocol_uses_official_single_grasp_contract(self):
        protocol = resolve_evaluation_protocol("realvlg_source")
        self.assertEqual(protocol.name, "realvlg_official")
        self.assertEqual(protocol.default_grasp_topk, (1,))
        self.assertEqual(protocol.grasp_evaluator, "realvlg_official")
        self.assertEqual(protocol.grasp_iou_threshold, 0.25)
        self.assertEqual(protocol.grasp_angle_threshold, 30.0)

    def test_public_executable_scene_ranges_are_preserved(self):
        self.assertEqual(list(REALVLG_EVAL_SCENES["seen"]), list(range(100, 130)))
        self.assertEqual(
            list(REALVLG_EVAL_SCENES["similar"]), list(range(130, 160))
        )
        self.assertEqual(list(REALVLG_EVAL_SCENES["novel"]), list(range(160, 190)))

    def test_identical_grasp_has_one_iou_and_is_correct(self):
        target = realvlg_rect_to_points8(50.0, 60.0, 20.0, 80.0)
        best_iou, best_angle, correct = evaluate_realvlg_grasp(
            [50.0, 60.0, 80.0, 999.0, 20.0], [target]
        )
        self.assertAlmostEqual(best_iou, 1.0, places=6)
        self.assertAlmostEqual(best_angle, 0.0, places=6)
        self.assertTrue(correct)

    def test_gripper_depth_is_fixed_and_thresholds_are_strict(self):
        target = realvlg_rect_to_points8(50.0, 60.0, 20.0, 80.0)
        best_iou, _best_angle, correct = evaluate_realvlg_grasp(
            [50.0, 60.0, 80.0, 400.0, 20.0], [target]
        )
        self.assertAlmostEqual(best_iou, 1.0, places=6)
        self.assertTrue(correct)

        # Horizontal 80x40 rectangles offset by 48 pixels have IoU 0.25.
        boundary = realvlg_rect_to_points8(98.0, 60.0, 0.0, 80.0)
        best_iou, _best_angle, correct = evaluate_realvlg_grasp(
            [50.0, 60.0, 80.0, 1.0, 0.0], [boundary]
        )
        self.assertAlmostEqual(best_iou, 0.25, places=6)
        self.assertFalse(correct)

    def test_angle_auto_detection_matches_public_implementation(self):
        # The official evaluator interprets two small values as radians.
        self.assertAlmostEqual(
            realvlg_angular_diff(1.0, 2.0),
            np.rad2deg(1.0),
            places=5,
        )

    def test_mask_metrics_match_public_formulas(self):
        target = np.zeros((4, 4), dtype=np.uint8)
        target[:2, :2] = 1
        self.assertAlmostEqual(realvlg_f_measure(target, target), 1.0, places=6)
        self.assertTrue(0.0 <= realvlg_s_measure(target, target) <= 1.0)
        self.assertAlmostEqual(realvlg_e_measure(target, target), 1.0, places=6)
        box = realvlg_mask_to_bbox(target)
        np.testing.assert_allclose(box, [0.0, 0.0, 2.0, 2.0])
        self.assertAlmostEqual(realvlg_giou(box, box), 1.0, places=6)
        self.assertAlmostEqual(realvlg_ciou(box, box), 1.0, places=6)

    def test_offset_v2_decoder_uses_grasp_relative_scale(self):
        loop = RealVLGValLoop.__new__(RealVLGValLoop)
        loop.cfg = SimpleNamespace(offset_resample_geometry=False)
        loop.width_factor = 100.0
        loop.offset_decode_mode = "grasp_relative"
        quality = np.zeros((32, 32), dtype=np.float32)
        quality[10, 10] = 1.0
        sine = np.zeros_like(quality)
        cosine = np.ones_like(quality)
        width = np.full_like(quality, 0.4)
        offset = np.zeros((2, 32, 32), dtype=np.float32)
        offset[0, 10, 10] = 0.5
        prediction = loop._decode_one_grasp(
            quality,
            sine,
            cosine,
            width,
            np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            scale=1.0,
            offset=offset,
        )
        expected_scale = np.hypot(40.0 * 0.25, 40.0 * 0.5)
        self.assertAlmostEqual(prediction[0], 10.0 + 0.5 * expected_scale)
        self.assertAlmostEqual(prediction[1], 10.0)


class RealVLGDatasetAdapterTest(unittest.TestCase):
    def _make_fixture(self, root):
        metadata_dir = root / "metadata" / "kinect" / "scene_0100"
        image_dir = root / "images"
        mask_dir = root / "masks"
        metadata_dir.mkdir(parents=True)
        image_dir.mkdir()
        mask_dir.mkdir()

        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[:, :, 1] = 127
        mask = np.zeros((40, 80), dtype=np.uint8)
        mask[10:30, 20:60] = 255
        self.assertTrue(cv2.imwrite(str(image_dir / "0000.png"), image))
        self.assertTrue(cv2.imwrite(str(mask_dir / "0000.png"), mask))
        grasp = realvlg_rect_to_points8(40.0, 20.0, 15.0, 30.0).tolist()
        payload = [
            {
                "image_name": "0000.png",
                "image_path": "images/0000.png",
                "object_id": "7",
                "mask_path": "masks/0000.png",
                "description": "grasp the green object near the center",
                "label": "green object",
                "bbox": [20, 10, 60, 30],
                "grasps": [grasp],
                "contact_points": [],
            }
        ]
        (metadata_dir / "0000.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        # Evaluation must ignore every frame except 0000.json.
        (metadata_dir / "0001.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_letterbox_and_original_coordinate_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_fixture(root)
            dataset = RealVLGDataset(
                root,
                input_size=64,
                split="seen",
                word_length=77,
                with_depth=False,
            )
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(tuple(sample["img"].shape), (3, 64, 64))
            self.assertEqual(tuple(sample["mask"].shape), (64, 64))
            self.assertEqual(tuple(sample["mask_original"].shape), (40, 80))
            self.assertEqual(sample["sample_id"], "scene_0100/0000.json#7")
            self.assertAlmostEqual(sample["scale"], 0.8, places=6)
            converted = realvlg_points8_to_rectangles(
                sample["grasps_points8"]
            )[0]
            np.testing.assert_allclose(
                converted[:5],
                [40.0, 20.0, 30.0, 40.0, 15.0],
                atol=1e-4,
            )

            offset_v2_dataset = RealVLGDataset(
                root,
                input_size=64,
                split="seen",
                word_length=77,
                with_depth=False,
                with_offset=True,
                offset_version="v2",
                offset_target_stride=4,
            )
            offset_masks = offset_v2_dataset[0]["grasp_masks"]
            self.assertEqual(tuple(offset_masks["off"].shape), (2, 16, 16))
            self.assertEqual(tuple(offset_masks["off_w"].shape), (1, 16, 16))
            self.assertGreater(float(offset_masks["off_w"].max()), 0.0)

    def test_profile_selects_official_evaluator(self):
        path = ROOT / "config" / "realvlg" / "drogoff.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        self.assertEqual(config["DATA"]["dataset"], "realvlg")
        self.assertEqual(config["DATA"]["dataset_args"]["train_fraction"], 0.1)
        self.assertEqual(config["TRAIN"]["epochs"], 10)
        self.assertEqual(config["TRAIN"]["word_len"], 77)
        self.assertFalse(config["TRAIN"]["amp"])
        self.assertEqual(config["TEST"]["val_loop"], "realvlg_val")
        self.assertEqual(
            config["TEST"]["evaluation_protocol"], "realvlg_official"
        )

    def test_offset_v2_profile_preserves_official_evaluator(self):
        path = ROOT / "config" / "realvlg" / "drogoff_offset_v2.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        self.assertEqual(config["DATA"]["offset_version"], "v2")
        self.assertEqual(config["MODEL"]["offset_head"], "lightweight")
        self.assertEqual(config["TEST"]["val_loop"], "realvlg_val")
        self.assertEqual(
            config["TEST"]["evaluation_protocol"], "realvlg_official"
        )
        self.assertEqual(
            config["TEST"]["offset_decode_mode"], "grasp_relative"
        )


if __name__ == "__main__":
    unittest.main()
