import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "toolrgs_runtime_device_test", ROOT / "toolrgs" / "runtime" / "device.py"
)
DEVICE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEVICE)


class OptimizerForeachTest(unittest.TestCase):
    def test_adam_accepts_explicit_foreach_disable(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        cfg = SimpleNamespace(
            optimizer="adam",
            base_lr=0.001,
            weight_decay=0.0,
            optimizer_foreach=False,
        )
        optimizer = DEVICE.build_optimizer([parameter], cfg)
        self.assertIs(optimizer.defaults["foreach"], False)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()

    def test_adam_preserves_default_when_option_is_absent(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        cfg = SimpleNamespace(
            optimizer="adam",
            base_lr=0.001,
            weight_decay=0.0,
        )
        optimizer = DEVICE.build_optimizer([parameter], cfg)
        self.assertIsNone(optimizer.defaults["foreach"])


if __name__ == "__main__":
    unittest.main()
