import importlib.util
from pathlib import Path
import unittest

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "toolrgs_native_dense", ROOT / "model" / "native_dense.py"
)
NATIVE_DENSE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE_DENSE)


class NativeDenseFusionTest(unittest.TestCase):
    def _module(self):
        return NATIVE_DENSE.NativeDenseFusionPyramid(
            in_dim=32,
            pyramid_dim=16,
            out_dim=24,
            text_dim=12,
            token_dim=8,
            dropout=0.0,
        )

    @staticmethod
    def _inputs():
        features = [
            torch.randn(2, 32, 8, 8, requires_grad=True)
            for _ in range(4)
        ]
        text = torch.randn(2, 6, 12, requires_grad=True)
        state = torch.randn(2, 12, requires_grad=True)
        padding = torch.tensor(
            [[False, False, False, True, True, True],
             [False, False, True, True, True, True]]
        )
        alignment = torch.randn(2, 1, 8, 8, requires_grad=True)
        return features, text, state, padding, alignment

    def test_output_shape_and_backward(self):
        module = self._module()
        inputs = self._inputs()
        output = module(*inputs)
        self.assertEqual(tuple(output.shape), (2, 24, 8, 8))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(inputs[0][0].grad)
        self.assertIsNotNone(inputs[1].grad)
        self.assertIsNotNone(inputs[4].grad)

    def test_padding_tokens_do_not_affect_output(self):
        module = self._module().eval()
        features, text, state, padding, alignment = self._inputs()
        changed = text.detach().clone()
        changed[padding] = torch.randn_like(changed[padding]) * 1000.0
        with torch.no_grad():
            reference = module(
                [value.detach() for value in features],
                text.detach(),
                state.detach(),
                padding,
                alignment.detach(),
            )
            actual = module(
                [value.detach() for value in features],
                changed,
                state.detach(),
                padding,
                alignment.detach(),
            )
        torch.testing.assert_close(reference, actual, atol=1e-5, rtol=1e-5)

    def test_v4_profile_and_decoder_removal_contract(self):
        config_path = ROOT / "config" / "realvlg" / "native_v4_drogoff.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["MODEL"]["architecture"], "drogoff")
        self.assertEqual(config["MODEL"]["native_variant"], "v4")
        self.assertEqual(config["TRAIN"]["epochs"], 36)
        self.assertEqual(config["TRAIN"]["val_start_epoch"], 11)
        self.assertEqual(config["TRAIN"]["val_freq"], 1)
        self.assertEqual(config["DATA"]["dataset_args"]["train_fraction"], 1.0)
        source = (ROOT / "model" / "drogoff.py").read_text(encoding="utf-8")
        self.assertIn("self.decoder = torch.nn.Identity()", source)
        self.assertIn("self.uses_query_decoder = False", source)


if __name__ == "__main__":
    unittest.main()
