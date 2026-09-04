from pathlib import Path
import unittest

from utils.config import load_cfg_from_cfg_file


ROOT = Path(__file__).resolve().parents[1]


class GraspAlignmentProfileTest(unittest.TestCase):
    def test_profiles_keep_offset_v1_and_start_without_checkpoints(self):
        expected = {
            "ocid_vlg.yaml": ("OCID-VLG", 17),
            "vcot.yaml": ("vcot", 17),
            "realvlg.yaml": ("realvlg", 77),
            "grasp_tools_v3.yaml": ("GraspTool", 32),
        }
        profile_root = ROOT / "config" / "experiments" / "grasp_alignment_v1"
        self.assertEqual(
            {path.name for path in profile_root.glob("*.yaml")}, set(expected)
        )
        for name, (dataset, word_len) in expected.items():
            cfg = load_cfg_from_cfg_file(profile_root / name)
            self.assertEqual(cfg.architecture, "drogoff")
            self.assertEqual(cfg.offset_version, "v1")
            self.assertEqual(cfg.dataset, dataset)
            self.assertEqual(cfg.word_len, word_len)
            self.assertEqual(cfg.fusion_adapter, "grasp_aware")
            self.assertTrue(cfg.grasp_alignment_enabled)
            self.assertEqual(cfg.visual_adapter_layer, [])
            self.assertEqual(cfg.txtual_adapter_layer, [])
            self.assertEqual(cfg.reciprocal_adapter_layers, [])
            self.assertEqual(cfg.object_alignment_loss_weight, 0.2)
            self.assertEqual(cfg.grasp_alignment_loss_weight, 0.1)
            self.assertEqual(cfg.region_text_contrastive_loss_weight, 0.05)
            self.assertIsNone(cfg.weight)
            self.assertIsNone(cfg.resume)


if __name__ == "__main__":
    unittest.main()
