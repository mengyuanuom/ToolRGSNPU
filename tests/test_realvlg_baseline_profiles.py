from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class RealVLGBaselineProfilesTest(unittest.TestCase):
    PROFILES = {
        "lgd.yaml": ("lgd", 224, "sigmoid"),
        "ggcnnclip.yaml": ("ggcnnclip", 416, "clamp"),
        "grconvnetclip.yaml": ("grconvnetclip", 416, "sigmoid"),
    }

    def test_full_graspnet_training_contract(self):
        for filename, (architecture, input_size, size_activation) in self.PROFILES.items():
            with self.subTest(profile=filename):
                path = ROOT / "config" / "realvlg" / filename
                profile = yaml.safe_load(path.read_text(encoding="utf-8"))
                data, model, train, test = (
                    profile["DATA"],
                    profile["MODEL"],
                    profile["TRAIN"],
                    profile["TEST"],
                )
                self.assertEqual(data["dataset"], "realvlg")
                self.assertEqual(data["dataset_args"]["camera_mode"], "kinect")
                self.assertEqual(data["dataset_args"]["train_fraction"], 1.0)
                self.assertEqual(data["grasp_size_factor"], 100.0)
                self.assertEqual(data["grasp_height"], 40.0)
                self.assertFalse(data["with_depth"])
                self.assertEqual(model["architecture"], architecture)
                self.assertEqual(train["input_size"], input_size)
                self.assertEqual(train["word_len"], 77)
                self.assertEqual(train["batch_size"], 256)
                self.assertEqual(train["epochs"], 24)
                self.assertEqual(train["val_start_epoch"], 11)
                self.assertEqual(train["val_freq"], 1)
                self.assertFalse(train["optimizer_foreach"])
                self.assertNotIn("predict_grasp_short_side", train)
                self.assertEqual(test["val_loop"], "realvlg_val")
                self.assertEqual(
                    test["evaluation_protocol"], "realvlg_official"
                )
                self.assertEqual(test["grasp_size_activation"], size_activation)


if __name__ == "__main__":
    unittest.main()
