from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DrogoffResourceProfileTest(unittest.TestCase):
    def test_all_drogoff_configs_use_conservative_global_defaults(self):
        paths = sorted((ROOT / "config").glob("*/drogoff*.yaml"))
        paths = [
            path for path in paths
            if path != ROOT / "config" / "vcot" / "drogoff_v2.yaml"
        ]
        self.assertEqual(len(paths), 3)
        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            self.assertEqual(cfg["MODEL"]["architecture"], "drogoff", path)
            train = cfg["TRAIN"]
            expected_batches = {
                "grasp_tools": (64, 32),
                "ocid_vlg": (24, 24),
                "vcot": (128, 8),
            }
            self.assertEqual(
                (train["batch_size"], train["batch_size_val"]),
                expected_batches[path.parent.name],
                path,
            )
            if path.parent.name == "vcot":
                self.assertEqual(train["epochs"], 36, path)
                self.assertEqual(train["milestones"], [30], path)
            self.assertEqual(train["workers"], 4, path)
            self.assertEqual(train["workers_val"], 2, path)
            self.assertEqual(train["print_freq"], 100, path)
            best_only = {"grasp_tools", "ocid_vlg"}
            expected_save_freq = 0 if path.parent.name in best_only else 5
            self.assertEqual(train["save_freq"], expected_save_freq, path)
            self.assertEqual(cfg["Distributed"]["dist_backend"], "hccl", path)
            self.assertTrue(cfg["TEST"]["offset_resample_geometry"], path)

    def test_vcot_offset_v2_profile_selects_new_contract(self):
        path = ROOT / "config" / "vcot" / "drogoff_v2.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        self.assertEqual(cfg["MODEL"]["architecture"], "drogoff")
        self.assertEqual(cfg["MODEL"]["offset_head"], "lightweight")
        self.assertEqual(cfg["DATA"]["offset_version"], "v2")
        self.assertEqual(cfg["DATA"]["offset_target_stride"], 4)
        self.assertEqual(cfg["TRAIN"]["offset_loss_weight"], 0.1)
        self.assertEqual(cfg["TEST"]["offset_decode_mode"], "grasp_relative")
        self.assertFalse(cfg["TEST"]["offset_resample_geometry"])


if __name__ == "__main__":
    unittest.main()
