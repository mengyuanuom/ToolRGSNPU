from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class GraspSizeProtocolTest(unittest.TestCase):
    def test_dataset_profiles_use_explicit_size_protocol(self):
        factors = {"grasp_tools": 300.0, "ocid_vlg": 100.0, "vcot": 300.0}
        for dataset, factor in factors.items():
            for path in (ROOT / "config" / dataset).glob("*.yaml"):
                config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
                data = config["DATA"]
                self.assertEqual(data["grasp_size_factor"], factor, path)
                self.assertEqual(data["grasp_height"], 20.0, path)
                expected_coordinate = "canvas" if dataset == "vcot" else "original"
                self.assertEqual(data["grasp_size_coordinate"], expected_coordinate, path)
                if dataset == "grasp_tools":
                    test = config["TEST"]
                    self.assertEqual(test["grasp_size_activation"], "auto", path)
                    self.assertEqual(test["grasp_iou_threshold"], 0.50, path)
                    self.assertEqual(
                        test["grasp_iou_thresholds"], [0.25, 0.50], path
                    )
                    self.assertEqual(test["grasp_angle_threshold"], 30.0, path)

    def test_source_contains_separate_raster_and_size_rectangles(self):
        source = (ROOT / "utils" / "dataset.py").read_text(encoding="utf-8")
        self.assertIn("size_rectangles=None", source)
        self.assertIn("target_width = size_rect[2]", source)
        self.assertIn("size_rectangles=grasp_target", source)

    def test_npu_runner_keeps_protocol_metadata_and_validation_warmup(self):
        source = (ROOT / "toolrgs" / "engine" / "runner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('checkpoint.get("grasp_size_factor")', source)
        self.assertIn('"grasp_size_coordinate": str(', source)
        self.assertIn('"grasp_size_activation": getattr(', source)
        self.assertIn('getattr(cfg, "val_start_epoch", 1)', source)

    def test_drogoff_uses_sigmoid_consistently_for_size_training(self):
        source = (ROOT / "model" / "drogoff.py").read_text(encoding="utf-8")
        self.assertIn('grasp_size_loss_activation = "sigmoid"', source)
        self.assertIn("torch.sigmoid(width), grasp_wid_mask", source)

    def test_crog_uses_sigmoid_consistently_for_size_training(self):
        source = (ROOT / "model" / "crog.py").read_text(encoding="utf-8")
        self.assertIn('grasp_size_loss_activation = "sigmoid"', source)
        self.assertIn(
            "torch.sigmoid(width), grasp_wid_mask", source
        )

    def test_all_vcot_profiles_resolve_size_activation_from_model(self):
        for path in (ROOT / "config" / "vcot").glob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            if config["TRAIN"]["predict_grasp_short_side"]:
                self.assertEqual(
                    config["TEST"]["grasp_size_activation"], "auto", path
                )

    def test_validation_reports_both_grasp_iou_thresholds(self):
        source = (ROOT / "toolrgs" / "engine" / "val_loop.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("grasp_metrics_by_iou", source)
        self.assertIn('f"J@{topk}(IoU={threshold})', source)


if __name__ == "__main__":
    unittest.main()
