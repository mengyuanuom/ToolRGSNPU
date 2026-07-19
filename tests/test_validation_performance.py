"""Source contracts for the optimized validation path."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ValidationPerformanceContractTest(unittest.TestCase):
    def test_dense_maps_use_one_host_transfer_and_one_topk_decode(self):
        source = (ROOT / "toolrgs" / "engine" / "val_loop.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertNotIn("def _sample_map", source)
        self.assertIn("torch.cat(dense_tensors, dim=1)", source)
        self.assertIn("num_grasps=self.max_topk", source)
        self.assertIn("rectangles[:topk]", source)

    def test_iou_does_not_allocate_full_image_masks(self):
        source = (ROOT / "utils" / "grasp_eval.py").read_text(
            encoding="utf-8-sig"
        )
        start = source.index("def calculate_iou(")
        end = source.index("def calculate_max_iou(", start)
        implementation = source[start:end]
        self.assertNotIn("np.zeros(shape)", implementation)
        self.assertIn("np.ravel_multi_index", implementation)
        self.assertIn("np.intersect1d", implementation)


if __name__ == "__main__":
    unittest.main()