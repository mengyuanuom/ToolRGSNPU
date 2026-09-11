import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

spec = importlib.util.spec_from_file_location('losses', Path(__file__).parents[1] / 'toolrgs/models/sigmoid_grasp_loss.py')
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)


class LossTests(unittest.TestCase):
    def test_quality_balances_background(self):
        x = torch.zeros(4, requires_grad=True)
        y = torch.tensor([1., 0., 0., 0.])
        loss = losses.quality_loss(x, y)
        self.assertAlmostEqual(loss.item(), 0.125)
        loss.backward()
        self.assertAlmostEqual(abs(x.grad[0].item()), abs(x.grad[1:].sum().item()))

    def test_empty_geometry(self):
        x = torch.ones(4, requires_grad=True)
        loss = losses.geometry_loss(x, torch.zeros(4), torch.zeros(4, dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(x.grad.abs().sum().item(), 0)

    def test_masked_background_gradient(self):
        x = torch.ones(2, requires_grad=True)
        losses.geometry_loss(x, torch.zeros(2), torch.tensor([True, False])).backward()
        self.assertGreater(x.grad[0].item(), 0)
        self.assertEqual(x.grad[1].item(), 0)

    def test_contract(self):
        m = SimpleNamespace(grasp_quality_activation='identity', grasp_size_loss_activation='clamp')
        losses.configure(m, SimpleNamespace())
        self.assertEqual(m.grasp_quality_activation, 'identity')
        losses.configure(m, SimpleNamespace(grasp_loss_profile='sigmoid_masked'))
        self.assertEqual(m.grasp_size_loss_activation, 'sigmoid')
        with self.assertRaises(ValueError):
            losses.configure(m, SimpleNamespace(grasp_loss_profile='sigmoid_masked', grasp_size_activation='clamp'))


if __name__ == '__main__':
    unittest.main()
