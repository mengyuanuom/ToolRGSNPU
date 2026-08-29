import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "toolrgs_native_adapter", ROOT / "model" / "native_adapter.py"
)
NATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE)
LoRALinear = NATIVE.LoRALinear
NativeFeaturePyramid = NATIVE.NativeFeaturePyramid
PaddingAwareCrossModalAdapter = NATIVE.PaddingAwareCrossModalAdapter
patch_text_alignment_loss = NATIVE.patch_text_alignment_loss


class NativeAdapterTest(unittest.TestCase):
    def test_lora_is_noop_at_initialization_and_trainable(self):
        torch.manual_seed(0)
        base = nn.Linear(12, 7)
        sample = torch.randn(2, 3, 12)
        expected = base(sample).detach()
        lora = LoRALinear(base, rank=3, alpha=6.0)
        torch.testing.assert_close(lora(sample), expected)
        self.assertFalse(lora.base.weight.requires_grad)
        self.assertTrue(lora.lora_down.weight.requires_grad)
        self.assertTrue(lora.lora_up.weight.requires_grad)

    def test_cross_modal_adapter_ignores_padding_values(self):
        torch.manual_seed(1)
        adapter = PaddingAwareCrossModalAdapter(32, 24, 16, 4).eval()
        visual = torch.randn(2, 9, 32)
        text = torch.randn(2, 6, 24)
        padding = torch.tensor(
            [[False, False, False, True, True, True],
             [False, False, True, True, True, True]]
        )
        altered = text.clone()
        altered[padding] = torch.randn_like(altered[padding]) * 1000.0
        torch.testing.assert_close(
            adapter(visual, text, padding),
            adapter(visual, altered, padding),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_native_pyramid_produces_decoder_grid(self):
        torch.manual_seed(2)
        pyramid = NativeFeaturePyramid(
            in_dim=48, pyramid_dim=16, out_dim=32, stages=4
        )
        features = [torch.randn(2, 48, 8, 8) for _ in range(4)]
        output = pyramid(features, torch.randn(2, 512))
        self.assertEqual(tuple(output.shape), (2, 32, 8, 8))

    def test_patch_text_alignment_loss_is_finite(self):
        logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 10:22] = 1.0
        loss = patch_text_alignment_loss(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)


class NativeV3ProfileTest(unittest.TestCase):
    def test_realvlg_native_v3_profile(self):
        path = ROOT / "config" / "realvlg" / "native_v3_drogoff.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        self.assertEqual(config["MODEL"]["architecture"], "drogoff")
        self.assertEqual(config["MODEL"]["native_variant"], "v3")
        self.assertEqual(config["DATA"]["offset_version"], "v2")
        self.assertEqual(config["DATA"]["dataset_args"]["train_fraction"], 1.0)
        train = config["TRAIN"]
        self.assertEqual(train["visual_adapter_layer"], [])
        self.assertEqual(train["txtual_adapter_layer"], [])
        self.assertEqual(train["native_visual_layers"], [2, 5, 8, 11])
        self.assertEqual(train["native_text_layers"], [2, 5, 8, 11])
        self.assertGreater(train["native_alignment_loss_weight"], 0.0)
        self.assertEqual(train["max_norm"], 1.0)
        self.assertFalse(train["optimizer_foreach"])
        self.assertEqual(train["val_start_epoch"], 11)
        self.assertEqual(train["val_freq"], 1)


if __name__ == "__main__":
    unittest.main()
