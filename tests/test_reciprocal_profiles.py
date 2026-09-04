from pathlib import Path
import unittest

from utils.config import load_cfg_from_cfg_file


ROOT = Path(__file__).resolve().parents[1]


class ReciprocalV1ProfileTest(unittest.TestCase):
    def test_all_three_profiles_keep_drogoff_v1_and_disable_legacy_adapters(self):
        expected = {
            "ocid_vlg.yaml": ("OCID-VLG", 17),
            "vcot.yaml": ("vcot", 17),
            "realvlg.yaml": ("realvlg", 77),
        }
        profile_root = ROOT / "config" / "experiments" / "reciprocal_v1"
        self.assertEqual(
            {path.name for path in profile_root.glob("*.yaml")},
            set(expected),
        )
        for name, (dataset, word_len) in expected.items():
            cfg = load_cfg_from_cfg_file(profile_root / name)
            self.assertEqual(cfg.architecture, "drogoff")
            self.assertEqual(cfg.offset_version, "v1")
            self.assertEqual(cfg.dataset, dataset)
            self.assertEqual(cfg.word_len, word_len)
            self.assertEqual(cfg.fusion_adapter, "reciprocal")
            self.assertEqual(cfg.visual_adapter_layer, [])
            self.assertEqual(cfg.txtual_adapter_layer, [])
            self.assertEqual(
                cfg.reciprocal_adapter_layers, [1, 3, 5, 7, 9, 11]
            )
            self.assertIsNone(cfg.weight)
            self.assertIsNone(cfg.resume)


if __name__ == "__main__":
    unittest.main()
