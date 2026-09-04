import importlib.util
from pathlib import Path
import sys
import unittest

from types import ModuleType

fvcore_module = ModuleType("fvcore")
fvcore_nn_module = ModuleType("fvcore.nn")
weight_init_module = ModuleType("fvcore.nn.weight_init")
weight_init_module.c2_xavier_fill = lambda module: None
fvcore_nn_module.weight_init = weight_init_module
fvcore_module.nn = fvcore_nn_module
sys.modules.setdefault("fvcore", fvcore_module)
sys.modules.setdefault("fvcore.nn", fvcore_nn_module)
sys.modules.setdefault("fvcore.nn.weight_init", weight_init_module)
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAYERS = _load_module("toolrgs_alignment_layers", ROOT / "model" / "layers.py")
MultiTaskProjector = LAYERS.MultiTaskProjector
OffsetMultiTaskProjector = LAYERS.OffsetMultiTaskProjector


class AlignmentProjectorGateTest(unittest.TestCase):
    def test_zero_initialized_gates_preserve_projector_outputs(self):
        torch.manual_seed(3)
        projector = MultiTaskProjector(
            word_dim=8,
            in_dim=4,
            kernel_size=3,
            use_alignment_gates=True,
        ).eval()
        features = torch.randn(2, 8, 3, 5)
        words = torch.randn(2, 8)
        object_gate = torch.randn(2, 1, 2, 3)
        grasp_gate = torch.randn(2, 1, 2, 3)
        baseline = projector(features, words)
        gated = projector(
            features,
            words,
            object_gate=object_gate,
            grasp_gate=grasp_gate,
        )
        for baseline_map, gated_map in zip(baseline, gated):
            torch.testing.assert_close(baseline_map, gated_map)

    def test_grasp_gate_skips_segmentation_branch(self):
        torch.manual_seed(4)
        projector = MultiTaskProjector(
            word_dim=8,
            in_dim=4,
            kernel_size=3,
            use_alignment_gates=True,
        ).eval()
        projector.grasp_gate_gain.data.fill_(1.0)
        features = torch.randn(1, 8, 3, 5)
        words = torch.randn(1, 8)
        grasp_gate = torch.full((1, 1, 3, 5), 8.0)
        baseline = projector(features, words)
        gated = projector(features, words, grasp_gate=grasp_gate)
        torch.testing.assert_close(baseline[0], gated[0])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(baseline[1:], gated[1:])))

    def test_offset_projector_accepts_both_alignment_maps(self):
        torch.manual_seed(5)
        projector = OffsetMultiTaskProjector(
            word_dim=8,
            in_dim=4,
            kernel_size=3,
            use_alignment_gates=True,
        ).eval()
        features = torch.randn(1, 8, 3, 5)
        words = torch.randn(1, 8)
        gate = torch.randn(1, 1, 3, 5)
        outputs = projector(features, words, object_gate=gate, grasp_gate=gate)
        self.assertEqual(len(outputs), 6)
        self.assertEqual(outputs[-1].shape[1], 2)
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs))


if __name__ == "__main__":
    unittest.main()
