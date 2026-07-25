from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "crog",
    "crogoff",
    "drog",
    "drogoff",
    "etrg",
    "etrg_r101",
    "ggcnnclip",
    "graspmamba",
    "grconvnetclip",
    "lgd",
    "maplegrasp",
}


class OCIDVLGResourceProfileTest(unittest.TestCase):
    def test_all_model_configs_use_complete_eight_npu_defaults(self):
        paths = sorted((ROOT / "config" / "ocid_vlg").glob("*.yaml"))
        self.assertEqual({path.stem for path in paths}, EXPECTED_MODELS)

        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            data = cfg["DATA"]
            train = cfg["TRAIN"]
            distributed = cfg["Distributed"]
            test = cfg["TEST"]

            self.assertEqual(data["root_path"], "./datasets/OCID-VLG", path)
            expected_word_len = 20 if path.stem.startswith("etrg") else 17
            self.assertEqual(train["word_len"], expected_word_len, path)
            self.assertFalse(train["amp"], path)
            self.assertFalse(train["sync_bn"], path)
            expected_batch = {"etrg": 10, "etrg_r101": 11}.get(path.stem, 24)
            self.assertEqual(train["batch_size"], expected_batch, path)
            self.assertEqual(train["batch_size_val"], expected_batch, path)
            self.assertEqual(train["base_lr"], 0.0001, path)
            expected_epochs = 40 if path.stem.startswith("etrg") else 50
            self.assertEqual(train["epochs"], expected_epochs, path)
            self.assertEqual(train["milestones"], [35], path)
            self.assertEqual(train["workers"], 4, path)
            self.assertEqual(train["workers_val"], 2, path)
            self.assertEqual(train["print_freq"], 100, path)
            self.assertEqual(train["save_freq"], 0, path)
            self.assertEqual(distributed["dist_url"], "env://", path)
            self.assertEqual(distributed["dist_backend"], "hccl", path)
            self.assertEqual(test["test_split"], "test", path)
            self.assertFalse(test["visualize"], path)

    def test_crog_uses_the_official_global_batch_and_learning_rate(self):
        path = ROOT / "config" / "ocid_vlg" / "crog.yaml"
        train = yaml.safe_load(path.read_text(encoding="utf-8-sig"))["TRAIN"]
        self.assertEqual(train["batch_size"], 24)
        self.assertEqual(train["batch_size"] // 8, 3)
        self.assertEqual(train["base_lr"], 0.0001)
        self.assertEqual(train["lr_multi"], 0.1)
        self.assertEqual(train["epochs"], 50)
        self.assertEqual(train["milestones"], [35])
        self.assertFalse(train["sync_bn"])


if __name__ == "__main__":
    unittest.main()
