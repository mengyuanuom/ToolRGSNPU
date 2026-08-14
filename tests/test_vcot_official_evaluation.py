from pathlib import Path
import unittest

import cv2
import numpy as np
import yaml

from toolrgs.evaluation import (
    calculate_vcot_grasp_success,
    resolve_evaluation_protocol,
    vcot_angle_within_threshold,
    vcot_rotated_iou,
)


ROOT = Path(__file__).resolve().parents[1]


class VCoTOfficialEvaluationTest(unittest.TestCase):
    def test_named_protocol_uses_single_prediction_contract(self):
        protocol = resolve_evaluation_protocol("vcot_source")
        self.assertEqual(protocol.name, "vcot_official")
        self.assertEqual(protocol.inverse_interpolation, cv2.INTER_NEAREST)
        self.assertEqual(protocol.grasp_canvas, (416, 416))
        self.assertEqual(protocol.default_grasp_topk, (1,))
        self.assertEqual(protocol.grasp_evaluator, "vcot_official")
        self.assertEqual(protocol.grasp_iou_threshold, 0.25)
        self.assertEqual(protocol.grasp_angle_threshold, 30.0)

    def test_angle_criterion_is_periodic_and_inclusive(self):
        self.assertTrue(vcot_angle_within_threshold(179.0, 1.0, 30.0))
        self.assertTrue(vcot_angle_within_threshold(0.0, 30.0, 30.0))
        self.assertFalse(vcot_angle_within_threshold(0.0, 30.01, 30.0))

    def test_iou_matches_continuous_rotated_rectangle_geometry(self):
        first = [50.0, 50.0, 40.0, 20.0, 15.0]
        second = [50.0, 50.0, 40.0, 20.0, 15.0]
        # OpenCV builds differ by roughly 1e-6 in polygon intersection area.
        self.assertAlmostEqual(vcot_rotated_iou(first, second), 1.0, places=5)

        # Two 40x20 horizontal rectangles offset by 24 pixels have IoU 0.25.
        boundary = [74.0, 50.0, 40.0, 20.0, 0.0]
        base = [50.0, 50.0, 40.0, 20.0, 0.0]
        self.assertAlmostEqual(vcot_rotated_iou(base, boundary), 0.25, places=6)
        self.assertEqual(
            calculate_vcot_grasp_success(base, [boundary], iou_threshold=0.25),
            1,
        )

    def test_official_success_preserves_ground_truth_width_and_height(self):
        prediction = [50.0, 50.0, 40.0, 20.0, 0.0]
        larger_target = [50.0, 50.0, 40.0, 40.0, 0.0, 0.0]
        self.assertAlmostEqual(
            vcot_rotated_iou(prediction, larger_target),
            0.5,
            places=6,
        )
        self.assertEqual(
            calculate_vcot_grasp_success(
                prediction,
                [larger_target],
                iou_threshold=0.75,
            ),
            0,
        )
        self.assertEqual(calculate_vcot_grasp_success(None, [larger_target]), 0)

    def test_every_vcot_npu_profile_selects_official_evaluation(self):
        paths = sorted((ROOT / "config" / "vcot").glob("*.yaml"))
        self.assertEqual(len(paths), 9)
        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            self.assertEqual(
                cfg["DATA"]["root_path"],
                "./datasets/graspanything-vcot",
                path,
            )
            self.assertEqual(
                cfg["DATA"]["split_root"],
                "./datasets/graspanything-vcot/split/vcot",
                path,
            )
            self.assertEqual(
                cfg["TEST"]["evaluation_protocol"],
                "vcot_official",
                path,
            )
            self.assertEqual(cfg["TEST"]["grasp_topk"], [1], path)
            self.assertEqual(cfg["TEST"]["grasp_iou_threshold"], 0.25, path)
            self.assertEqual(cfg["TEST"]["grasp_angle_threshold"], 30.0, path)
            self.assertFalse(cfg["TEST"]["filter_grasps_by_segmentation"], path)
            self.assertEqual(cfg["DATA"]["train_split"], "train_official", path)
            self.assertEqual(cfg["DATA"]["val_split"], "val_official", path)
            self.assertEqual(cfg["DATA"]["grasp_size_factor"], 416, path)
            self.assertEqual(cfg["DATA"]["grasp_size_coordinate"], "original", path)
            self.assertEqual(cfg["DATA"]["grasp_target_policy"], "first", path)
            self.assertEqual(cfg["DATA"]["vcot_official_val_size"], 5000, path)


if __name__ == "__main__":
    unittest.main()
