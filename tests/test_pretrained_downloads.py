"""Contracts for the documented pretrained-weight manifest."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "download_pretrained.py"


def load_downloader():
    spec = importlib.util.spec_from_file_location("download_pretrained", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PretrainedDownloadManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_downloader()

    def test_manifest_covers_every_configured_backbone_filename(self):
        expected = {
            "RN50.pt",
            "RN101.pt",
            "ViT-B-16.pt",
            "dinov2_vitb14_reg4_pretrain.pth",
            "mambavision_tiny_1k.pth.tar",
            "resnet18-f37072fd.pth",
        }
        actual = {item.filename for item in self.module.ARTIFACTS.values()}
        self.assertEqual(actual, expected)

    def test_every_artifact_uses_https_and_a_safe_filename(self):
        for name, artifact in self.module.ARTIFACTS.items():
            with self.subTest(name=name):
                self.assertTrue(artifact.url.startswith("https://"))
                self.assertEqual(Path(artifact.filename).name, artifact.filename)
                self.assertNotIn("..", artifact.filename)

    def test_official_clip_urls_have_full_checksum(self):
        for name in ("clip-rn50", "clip-rn101", "clip-vit-b16"):
            artifact = self.module.ARTIFACTS[name]
            with self.subTest(name=name):
                self.assertEqual(len(artifact.sha256), 64)
                self.assertIn(artifact.sha256, artifact.url)


if __name__ == "__main__":
    unittest.main()
