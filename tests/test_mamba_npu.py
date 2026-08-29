import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import yaml

from model.mamba_npu import (
    _depthwise_conv1d_as_conv2d,
    install_mamba_ssm_npu_shim,
    patch_mambavision_for_npu,
    selective_scan_fn,
)


ROOT = Path(__file__).resolve().parents[1]


class MambaVisionMixer(nn.Module):
    """Tiny parameter-compatible mixer used to test the bound NPU forward."""

    def __init__(self):
        super().__init__()
        self.d_model = 4
        self.d_state = 2
        self.d_conv = 4
        self.d_inner = 4
        self.dt_rank = 1
        self.in_proj = nn.Linear(4, 4, bias=False)
        self.x_proj = nn.Linear(2, 5, bias=False)
        self.dt_proj = nn.Linear(1, 2, bias=True)
        self.A_log = nn.Parameter(torch.zeros(2, 2))
        self.D = nn.Parameter(torch.ones(2))
        self.out_proj = nn.Linear(4, 4, bias=False)
        self.conv1d_x = nn.Conv1d(2, 2, 4, groups=2, bias=False)
        self.conv1d_z = nn.Conv1d(2, 2, 4, groups=2, bias=False)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_attn = True


class TinyMambaVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = MambaVisionMixer()
        self.attention = Attention()


class MambaNPUOperatorTest(unittest.TestCase):
    def _scan_inputs(self):
        torch.manual_seed(23)
        u = torch.randn(2, 3, 7, requires_grad=True)
        delta = torch.randn(2, 3, 7, requires_grad=True)
        A = (-torch.rand(3, 2)).requires_grad_()
        B = torch.randn(2, 2, 7, requires_grad=True)
        C = torch.randn(2, 2, 7, requires_grad=True)
        D = torch.randn(3, requires_grad=True)
        return u, delta, A, B, C, D

    def test_parallel_scan_matches_sequential_reference(self):
        inputs = self._scan_inputs()
        expected = selective_scan_fn(
            *inputs,
            delta_softplus=True,
            backend="sequential",
            checkpoint_scan=False,
        )
        actual = selective_scan_fn(
            *inputs,
            delta_softplus=True,
            backend="parallel",
            checkpoint_scan=False,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_even_depthwise_kernel_matches_same_padding(self):
        torch.manual_seed(29)
        convolution = nn.Conv1d(3, 3, 4, groups=3, bias=True)
        inputs = torch.randn(2, 3, 9)
        expected = torch.nn.functional.conv1d(
            inputs,
            convolution.weight,
            convolution.bias,
            padding="same",
            groups=convolution.groups,
        )
        actual = _depthwise_conv1d_as_conv2d(inputs, convolution)
        self.assertEqual(tuple(actual.shape), tuple(inputs.shape))
        torch.testing.assert_close(actual, expected)

    def test_parallel_scan_has_finite_gradients(self):
        inputs = self._scan_inputs()
        output = selective_scan_fn(
            *inputs,
            delta_softplus=True,
            backend="parallel",
            checkpoint_scan=True,
        )
        output.square().mean().backward()
        for value in inputs:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_import_shim_exposes_selective_scan_contract(self):
        with mock.patch.dict(sys.modules, clear=False):
            install_mamba_ssm_npu_shim()
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as fn

            self.assertTrue(callable(fn))

    def test_model_patch_preserves_parameters_and_replaces_forwards(self):
        model = TinyMambaVision()
        state_keys = tuple(model.state_dict())
        report = patch_mambavision_for_npu(
            model, scan_backend="parallel", checkpoint_scan=False
        )
        self.assertEqual(report.mixers, 1)
        self.assertEqual(report.attentions, 1)
        self.assertFalse(model.attention.fused_attn)
        self.assertEqual(tuple(model.state_dict()), state_keys)

        inputs = torch.randn(2, 5, 4, requires_grad=True)
        output = model.mixer(inputs)
        self.assertEqual(tuple(output.shape), (2, 5, 4))
        output.mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_all_graspmamba_profiles_enable_portable_npu_scan(self):
        for directory in ("grasp_tools", "ocid_vlg", "vcot"):
            path = ROOT / "config" / directory / "graspmamba.yaml"
            train = yaml.safe_load(path.read_text(encoding="utf-8-sig"))["TRAIN"]
            self.assertTrue(train["mamba_npu_fallback"], path)
            self.assertEqual(train["mamba_npu_scan_backend"], "parallel", path)
            self.assertTrue(train["mamba_npu_scan_checkpoint"], path)
            self.assertFalse(train["amp"], path)
            self.assertFalse(train["sync_bn"], path)

    def test_install_script_never_resolves_cuda_mamba_dependencies(self):
        source = (ROOT / "tools" / "install_graspmamba_npu.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("pip install --no-deps -r requirement-mamba.txt", source)
        self.assertIn("pip install --no-deps mambavision==1.2.0", source)
        requirements = (ROOT / "requirement-mamba.txt").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("mamba-ssm==", requirements)


if __name__ == "__main__":
    unittest.main()
