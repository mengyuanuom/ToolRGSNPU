from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DrogoffResourceProfileTest(unittest.TestCase):
    def test_all_drogoff_configs_use_conservative_global_defaults(self):
        paths = sorted((ROOT / "config").glob("*/drogoff*.yaml"))
        paths = [
            path for path in paths
            if path.name != "drogoff_offset_v2.yaml"
        ]
        self.assertEqual(len(paths), 4)
        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            self.assertEqual(cfg["MODEL"]["architecture"], "drogoff", path)
            self.assertEqual(cfg["DATA"]["offset_version"], "v1", path)
            train = cfg["TRAIN"]
            expected_batches = {
                "grasp_tools": (64, 32),
                "ocid_vlg": (24, 24),
                "realvlg": (128, 32),
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
            best_only = {"grasp_tools", "ocid_vlg", "realvlg"}
            expected_save_freq = 0 if path.parent.name in best_only else 5
            self.assertEqual(train["save_freq"], expected_save_freq, path)
            self.assertEqual(cfg["Distributed"]["dist_backend"], "hccl", path)
            self.assertTrue(cfg["TEST"]["offset_resample_geometry"], path)

    def test_every_dataset_exposes_dense_offset_v2(self):
        paths = sorted((ROOT / "config").glob("*/drogoff_offset_v2.yaml"))
        self.assertEqual(
            {path.parent.name for path in paths},
            {"grasp_tools", "ocid_vlg", "realvlg", "vcot"},
        )
        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            self.assertEqual(cfg["MODEL"]["architecture"], "drogoff", path)
            self.assertEqual(cfg["MODEL"]["offset_head"], "lightweight", path)
            self.assertEqual(cfg["MODEL"]["offset_hidden_dim"], 64, path)
            self.assertEqual(cfg["DATA"]["offset_version"], "v2", path)
            self.assertEqual(cfg["DATA"]["offset_target_stride"], 4, path)
            self.assertEqual(cfg["DATA"]["offset_weight_floor"], 0.25, path)
            self.assertEqual(cfg["TRAIN"]["offset_loss_weight"], 0.1, path)
            self.assertEqual(
                cfg["TEST"]["offset_decode_mode"], "grasp_relative", path
            )
            self.assertFalse(cfg["TEST"]["offset_resample_geometry"], path)


if __name__ == "__main__":
    unittest.main()
