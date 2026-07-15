import unittest

import numpy as np

from toolrgs.engine.hooks import Hook, HookList, LoopState
from toolrgs.evaluation import (
    BinarySegmentationMetric,
    DenseGraspPostProcessor,
    GraspSuccessMetric,
    corners_to_five,
    five_to_corners,
    inverse_warp,
    refine_with_offset,
    targets_to_six,
)
from toolrgs.registry import HOOKS, METRICS, POSTPROCESSORS


class EvaluationComponentTest(unittest.TestCase):
    def test_rotated_rectangle_round_trip_preserves_geometry(self):
        rectangle = np.array([32.0, 24.0, 18.0, 8.0, 27.0], dtype=np.float32)
        restored = corners_to_five(five_to_corners(rectangle))
        np.testing.assert_allclose(restored[:4], rectangle[:4], atol=1e-4)
        self.assertAlmostEqual(abs(restored[4]), abs(rectangle[4]), places=4)

    def test_inverse_warp_and_offset_refinement_share_coordinate_contract(self):
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        source = np.arange(16, dtype=np.float32).reshape(4, 4)
        np.testing.assert_array_equal(inverse_warp(source, identity, (4, 4)), source)

        offset = np.zeros((2, 32, 32), dtype=np.float32)
        offset[0] = 0.5
        offset[1] = -0.25
        refined = refine_with_offset(
            [[10.0, 20.0, 8.0, 4.0, 0.0]], offset, identity, radius=20
        )
        np.testing.assert_allclose(refined[0][:2], [20.0, 15.0], atol=1e-4)
        np.testing.assert_allclose(refined[0][2:4], [8.0, 4.0], atol=1e-4)

    def test_target_adapter_keeps_six_values_and_expands_other_formats(self):
        six = np.array([10, 20, 30, 40, 50, 7], dtype=np.float32)
        converted = targets_to_six([six, [10, 20, 30, 40, 50]])
        np.testing.assert_array_equal(converted[0], six)
        np.testing.assert_array_equal(
            converted[1], np.array([10, 20, 30, 40, 50, 0], dtype=np.float32)
        )

    def test_dense_grasp_postprocessor_decodes_peak_angle_and_width(self):
        quality = np.zeros((8, 8), dtype=np.float32)
        quality[3, 4] = 0.9
        sine = np.ones_like(quality)
        cosine = np.zeros_like(quality)
        width = np.full_like(quality, 0.5)
        processor = DenseGraspPostProcessor(num_grasps=1)

        detections = processor(
            quality, sine, cosine, width, spatial_scale=2.0
        )
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual((detection.x, detection.y), (4.0, 3.0))
        self.assertAlmostEqual(detection.angle_degrees, 45.0)
        self.assertAlmostEqual(detection.width, 100.0)
        self.assertAlmostEqual(detection.height, 40.0)

    def test_binary_segmentation_metric_uses_per_sample_iou(self):
        prediction = np.array(
            [[[1, 1], [0, 0]], [[1, 0], [0, 0]]], dtype=np.float32
        )
        target = np.array(
            [[[1, 0], [0, 0]], [[1, 0], [0, 0]]], dtype=np.uint8
        )
        metric = BinarySegmentationMetric(iou_thresholds=(0.5, 0.9))
        metric.update(prediction, target)
        result = metric.compute()
        self.assertAlmostEqual(result["iou"], 0.75, places=5)
        self.assertEqual(result["precision"], {"Pr@50": 0.5, "Pr@90": 0.5})

    def test_grasp_success_metric_aggregates_each_topk(self):
        metric = GraspSuccessMetric(topk=(1, 5))
        metric.update(1, True)
        metric.update(1, False)
        metric.update(5, True)
        self.assertEqual(metric.compute(), {"J@1": 0.5, "J@5": 1.0})

    def test_evaluation_components_are_registered(self):
        self.assertIn("binary_segmentation", METRICS)
        self.assertIn("grasp_success", METRICS)
        self.assertIn("dense_grasp", POSTPROCESSORS)


class HookLifecycleTest(unittest.TestCase):
    def test_hooks_run_in_priority_order(self):
        calls = []

        class LateHook(Hook):
            priority = 80

            def before_epoch(self, loop, state):
                calls.append("late")

        class EarlyHook(Hook):
            priority = 10

            def before_epoch(self, loop, state):
                calls.append("early")

        HookList([LateHook(), EarlyHook()]).call(
            "before_epoch", loop=object(), state=LoopState(epoch=1)
        )
        self.assertEqual(calls, ["early", "late"])
        self.assertIn("noop", HOOKS)


if __name__ == "__main__":
    unittest.main()
