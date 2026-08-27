from pathlib import Path
import unittest

import torch

from model.layers import OffsetMultiTaskProjector
from model.offset_transport import OffsetGuidedFeatureTransport
from model.transport_projector import OffsetTransportProjector
from utils.config import load_cfg_from_cfg_file


ROOT = Path(__file__).resolve().parents[1]


class OffsetTransportTest(unittest.TestCase):
    def _projector(self):
        torch.manual_seed(7)
        baseline = OffsetMultiTaskProjector(
            word_dim=16,
            in_dim=8,
            offset_head="lightweight",
            offset_hidden_dim=16,
        ).eval()
        return OffsetTransportProjector(
            baseline,
            hidden_dim=16,
            max_displacement=2.0,
        ).eval()

    def test_zero_gate_is_baseline_equivalent(self):
        projector = self._projector()
        features = torch.randn(2, 16, 4, 4)
        text = torch.randn(2, 16)

        expected = (*projector.base(features, text), projector.offset(features))
        actual = projector(features, text)
        self.assertEqual(len(actual), 6)
        for baseline, transported in zip(expected, actual):
            torch.testing.assert_close(transported, baseline)

    def test_segmentation_stays_on_original_path_after_gate_opens(self):
        projector = self._projector()
        projector.transport.branch_gate.data.fill_(0.5)
        features = torch.randn(1, 16, 4, 4)
        text = torch.randn(1, 16)

        baseline_segmentation = projector.base(features, text)[0]
        transported_segmentation = projector(features, text)[0]
        torch.testing.assert_close(
            transported_segmentation, baseline_segmentation
        )

    def test_offset_direction_samples_predicted_center(self):
        transport = OffsetGuidedFeatureTransport(
            channels=1, branches=1, hidden_dim=16, max_displacement=1.0
        )
        features = torch.arange(5.0).reshape(1, 1, 1, 5)
        offset = torch.zeros(1, 2, 1, 5)
        offset[:, 0] = 1.0

        sampled = transport.sample_center_context(features, offset)
        torch.testing.assert_close(
            sampled.flatten(), torch.tensor([1.0, 2.0, 3.0, 4.0, 4.0])
        )

    def test_zero_gate_receives_learning_signal(self):
        projector = self._projector().train()
        features = torch.randn(2, 16, 4, 4)
        text = torch.randn(2, 16)
        loss = sum(output.mean() for output in projector(features, text)[1:5])
        loss.backward()

        gradient = projector.transport.branch_gate.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_ocid_profile_enables_only_transport_extension(self):
        cfg = load_cfg_from_cfg_file(
            ROOT / "config" / "experiments" / "ocid_vlg"
            / "drogoff_transport.yaml"
        )
        self.assertEqual(cfg.architecture, "drogoff")
        self.assertEqual(cfg.offset_version, "v2")
        self.assertTrue(cfg.offset_transport_enabled)
        self.assertEqual(cfg.offset_transport_hidden_dim, 64)
        self.assertEqual(cfg.offset_transport_max_displacement, 6.0)
        self.assertEqual(cfg.visual_adapter_layer, [1, 3, 5, 7, 9, 11])
        self.assertFalse(hasattr(cfg, "native_variant"))


if __name__ == "__main__":
    unittest.main()
