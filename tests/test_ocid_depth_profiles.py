"""Keep RGB-only OCID-VLG experiments independent from depth files."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class OCIDDepthProfileTest(unittest.TestCase):
    def test_only_etrg_configs_request_depth(self):
        for path in (ROOT / "config" / "ocid_vlg").glob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            expected = path.stem.startswith("etrg")
            with self.subTest(config=path.name):
                self.assertEqual(bool(config["DATA"]["with_depth"]), expected)


if __name__ == "__main__":
    unittest.main()
