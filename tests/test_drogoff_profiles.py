from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DrogoffResourceProfileTest(unittest.TestCase):
    def test_all_drogoff_configs_use_conservative_two_npu_defaults(self):
        paths = sorted((ROOT / "config").glob("*/drogoff.yaml"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            self.assertEqual(cfg["MODEL"]["architecture"], "drogoff", path)
            train = cfg["TRAIN"]
            self.assertEqual(train["batch_size"], 8, path)
            self.assertEqual(train["batch_size_val"], 4, path)
            self.assertEqual(train["workers"], 4, path)
            self.assertEqual(train["workers_val"], 2, path)
            self.assertEqual(train["print_freq"], 100, path)
            self.assertEqual(train["save_freq"], 5, path)
            self.assertEqual(cfg["Distributed"]["dist_backend"], "hccl", path)


if __name__ == "__main__":
    unittest.main()
