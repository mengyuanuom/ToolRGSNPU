import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ALIGNMENT = _load_module(
    "toolrgs_grasp_alignment", ROOT / "model" / "grasp_alignment.py"
)
HierarchicalDinoClipAlignment = ALIGNMENT.HierarchicalDinoClipAlignment
LocalTopKAffinity = ALIGNMENT.LocalTopKAffinity
hierarchical_alignment_losses = ALIGNMENT.hierarchical_alignment_losses


class GraspAlignmentTest(unittest.TestCase):
    def test_dual_alignment_maps_are_finite_and_differentiable(self):
        torch.manual_seed(1)
        module = HierarchicalDinoClipAlignment(
            visual_dim=32,
            text_dim=24,
            hidden_dim=16,
            stages=3,
            text_heads=4,
            affinity_topk=4,
        )
        visual = [
            torch.randn(2, 32, 4, 6, requires_grad=True)
            for _ in range(3)
        ]
        text = torch.randn(2, 7, 24, requires_grad=True)
        state = torch.randn(2, 24, requires_grad=True)
        padding = torch.tensor(
            [
                [False, False, False, False, True, True, True],
                [False, False, False, True, True, True, True],
            ]
        )
        output = module(visual, text, state, padding)
        self.assertEqual(
            set(output),
            {
                "object_alignment", "grasp_alignment", "alignment_features",
                "object_query", "grasp_query",
            },
        )
        self.assertEqual(tuple(output["object_alignment"].shape), (2, 1, 4, 6))
        self.assertEqual(tuple(output["grasp_alignment"].shape), (2, 1, 4, 6))
        loss = output["object_alignment"].mean() + output["grasp_alignment"].mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(value.grad is not None for value in visual))
        self.assertIsNotNone(text.grad)
        self.assertIsNotNone(state.grad)

    def test_padding_content_does_not_change_alignment(self):
        torch.manual_seed(2)
        module = HierarchicalDinoClipAlignment(
            visual_dim=16,
            text_dim=12,
            hidden_dim=8,
            stages=2,
            text_heads=2,
            dropout=0.0,
        ).eval()
        visual = [torch.randn(1, 16, 3, 5) for _ in range(2)]
        text = torch.randn(1, 6, 12)
        altered = text.clone()
        padding = torch.tensor([[False, False, False, True, True, True]])
        altered[padding] = torch.randn_like(altered[padding]) * 1000.0
        state = torch.randn(1, 12)
        output_a = module(visual, text, state, padding)
        output_b = module(visual, altered, state, padding)
        for key in output_a:
            torch.testing.assert_close(output_a[key], output_b[key])

    def test_grasp_targets_and_all_auxiliary_losses_are_finite(self):
        object_logits = torch.randn(2, 1, 5, 7, requires_grad=True)
        grasp_logits = torch.randn(2, 1, 5, 7, requires_grad=True)
        mask = torch.zeros(2, 1, 20, 28)
        quality = torch.zeros_like(mask)
        center = torch.zeros_like(mask)
        mask[:, :, 4:17, 5:23] = 1.0
        quality[:, :, 8:15, 10:20] = 1.0
        center[:, :, 10:13, 13:17] = 1.0
        alignment_features = torch.randn(2, 8, 5, 7, requires_grad=True)
        object_query = torch.randn(2, 8, requires_grad=True)
        text_ids = torch.tensor([[1, 2, 3, 0], [1, 2, 3, 0]])
        losses = hierarchical_alignment_losses(
            object_logits,
            grasp_logits,
            mask,
            quality,
            center,
            alignment_features=alignment_features,
            object_query=object_query,
            text_ids=text_ids,
        )
        self.assertEqual(
            set(losses),
            {"object", "contrastive", "grasp", "ranking", "subset", "target"},
        )
        self.assertTrue(all(torch.isfinite(losses[key]) for key in losses if key != "target"))
        self.assertGreater(losses["target"].sum().item(), 0.0)
        total = sum(
            losses[key]
            for key in ("object", "contrastive", "grasp", "ranking", "subset")
        )
        total.backward()
        self.assertIsNotNone(object_logits.grad)
        self.assertIsNotNone(grasp_logits.grad)
        self.assertIsNotNone(alignment_features.grad)
        self.assertIsNotNone(object_query.grad)

    def test_affinity_preserves_constant_logits(self):
        module = LocalTopKAffinity(kernel_size=3, topk=4, steps=2, blend=0.5)
        features = torch.randn(2, 8, 4, 5)
        logits = torch.full((2, 1, 4, 5), 2.0)
        torch.testing.assert_close(module(features, logits), logits)


if __name__ == "__main__":
    unittest.main()
