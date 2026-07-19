from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "crog",
    "crogoff",
    "drog",
    "drogoff",
    "drogoff_v2",
    "ggcnnclip",
    "graspmamba",
    "grconvnetclip",
    "lgd",
    "maplegrasp",
}


class GraspToolResourceProfileTest(unittest.TestCase):
    def test_all_model_configs_use_complete_v2_eight_npu_defaults(self):
        paths = sorted((ROOT / "config" / "grasp_tools").glob("*.yaml"))
        self.assertEqual({path.stem for path in paths}, EXPECTED_MODELS)

        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            data = cfg["DATA"]
            train = cfg["TRAIN"]
            distributed = cfg["Distributed"]
            test = cfg["TEST"]

            self.assertEqual(
                data["root_path"], "./datasets/grasp-tools/aug_graspall_v2", path
            )
            self.assertEqual(train["word_len"], 32, path)
            self.assertFalse(train["amp"], path)
            self.assertFalse(train["sync_bn"], path)
            self.assertEqual(train["workers"], 4, path)
            self.assertEqual(train["workers_val"], 2, path)
            self.assertEqual(train["print_freq"], 100, path)
            self.assertEqual(train["save_freq"], 0, path)
            self.assertEqual(distributed["dist_url"], "env://", path)
            self.assertEqual(distributed["dist_backend"], "hccl", path)
            self.assertEqual(test["test_split"], "test", path)
            self.assertFalse(test["visualize"], path)


if __name__ == "__main__":
    unittest.main()
