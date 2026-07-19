"""Keep RGB-only OCID-VLG experiments independent from depth files."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class OCIDDepthProfileTest(unittest.TestCase):
    def test_default_profiles_are_rgb_only(self):
        for path in (ROOT / "config" / "ocid_vlg").glob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))

            with self.subTest(config=path.name):
                self.assertFalse(bool(config["DATA"]["with_depth"]))
                if path.stem.startswith("etrg"):
                    self.assertEqual(config["TRAIN"]["etrg_input_mode"], "rgb")


if __name__ == "__main__":
    unittest.main()
