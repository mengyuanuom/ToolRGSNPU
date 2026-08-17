import unittest
from pathlib import Path
import sys
import types

import numpy as np
import yaml
sys.modules.setdefault("lmdb", types.ModuleType("lmdb"))
sys.modules.setdefault("pyarrow", types.ModuleType("pyarrow"))


ftfy_stub = types.ModuleType("ftfy")
ftfy_stub.fix_text = lambda value: value
sys.modules.setdefault("ftfy", ftfy_stub)
regex_stub = types.ModuleType("regex")
regex_stub.IGNORECASE = 0
regex_stub.compile = lambda pattern, flags=0: pattern
regex_stub.findall = lambda pattern, value: []
sys.modules.setdefault("regex", regex_stub)
from toolrgs.evaluation.postprocessors import DenseGraspPostProcessor
from toolrgs.structures import GraspOutput
from utils.dataset import GraspTransforms


ROOT = Path(__file__).resolve().parents[1]



class VCoTShortSideTests(unittest.TestCase):
    def test_all_vcot_grasp_profiles_enable_short_side_regression(self):
        paths = sorted((ROOT / "config" / "vcot").glob("*.yaml"))
        self.assertEqual(len(paths), 12)
        for path in paths:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if path.name == "maplegrasp_stage1.yaml":
                self.assertFalse(data["TRAIN"]["predict_grasp_short_side"], path)
            else:
                self.assertTrue(data["TRAIN"]["predict_grasp_short_side"], path)
                self.assertEqual(data["TRAIN"]["short_side_loss_weight"], 1.0, path)

    def test_short_side_mask_generation_is_vcot_opt_in(self):
        transform = GraspTransforms(width_factor=100, width=32, height=32)
        rectangle = np.array([[16.0, 16.0, 8.0, 4.0, 0.0, 0.0]], dtype=np.float32)
        masks = transform.generate_masks(rectangle)
        self.assertNotIn("short", masks)

    def test_original_short_side_is_encoded_independently_of_canvas_geometry(self):
        transform = GraspTransforms(
            width_factor=100, width=32, height=32, predict_short_side=True
        )
        canvas = np.array([[16.0, 16.0, 8.0, 4.0, 0.0, 0.0]], dtype=np.float32)
        original = np.array([[20.0, 20.0, 80.0, 30.0, 0.0, 0.0]], dtype=np.float32)
        masks = transform.generate_masks(canvas, size_rectangles=original)
        self.assertIn("short", masks)
        width_peak = float(masks["wid"].max())
        short_peak = float(masks["short"].max())
        self.assertGreater(short_peak, 0.0)
        # Gaussian smoothing scales both maps equally; the 80:30 source ratio remains.
        self.assertAlmostEqual(width_peak / short_peak, 80.0 / 30.0, places=1)

    def test_all_grasp_models_use_native_short_side_heads(self):
        builder = (ROOT / "model" / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("ShortSideRegressionAdapter", builder)
        sources = (
            ROOT / "model" / "crog.py",
            ROOT / "model" / "crogoff.py",
            ROOT / "model" / "drog.py",
            ROOT / "model" / "drogoff.py",
            ROOT / "model" / "ggcnnclip.py",
            ROOT / "model" / "grconvnetclip.py",
            ROOT / "model" / "graspmamba.py",
            ROOT / "model" / "lgd.py",
            ROOT / "model" / "maplegrasp.py",
            ROOT / "model" / "etrg" / "model.py",
        )
        for path in sources:
            source = path.read_text(encoding="utf-8")
            with self.subTest(model=path.name):
                self.assertIn("predicts_grasp_short_side", source)
                self.assertIn("grasp_short_mask", source)
                self.assertIn('"sigmoid"', source)

    def test_postprocessor_uses_predicted_short_side(self):
        quality = np.zeros((9, 9), dtype=np.float32)
        quality[4, 4] = 1.0
        sine = np.zeros_like(quality)
        cosine = np.ones_like(quality)
        width = np.full_like(quality, 0.5)
        short = np.full_like(quality, 0.25)
        processor = DenseGraspPostProcessor(width_factor=100.0, grasp_height=20.0)
        detection = processor(
            quality, sine, cosine, width, short_side=short, num_grasps=1
        )[0]
        self.assertEqual(detection.width, 50.0)
        self.assertEqual(detection.height, 25.0)

    def test_named_output_keeps_offset_and_short_side_unambiguous(self):
        output = GraspOutput(1, 2, 3, 4, 5, offset=6, short_side=7)
        self.assertEqual(output.as_tuple(), (1, 2, 3, 4, 5, 7, 6))
        restored = GraspOutput.from_legacy(output.as_tuple())
        self.assertEqual(restored.offset, 6)
        self.assertEqual(restored.short_side, 7)


if __name__ == "__main__":
    unittest.main()
