import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ADAPTER = _load_module(
    "toolrgs_v1_adapter", ROOT / "model" / "adapter.py"
)
ReciprocalVisionLanguageAdapter = ADAPTER.ReciprocalVisionLanguageAdapter


def _load_fusion_module():
    package_name = "toolrgs_v1_model"
    package = ModuleType(package_name)
    package.__path__ = [str(ROOT / "model")]
    sys.modules[package_name] = package

    adapter_module = _load_module(
        f"{package_name}.adapter", ROOT / "model" / "adapter.py"
    )
    layers_module = ModuleType(f"{package_name}.layers")
    layers_module.conv_layer = lambda *args, **kwargs: nn.Identity()
    layers_module.deconv_layer = lambda *args, **kwargs: nn.Identity()
    sys.modules[layers_module.__name__] = layers_module
    return _load_module(
        f"{package_name}.fusion", ROOT / "model" / "fusion.py"
    )


FUSION = _load_fusion_module()


class _TextBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(512, 512)

    def forward(self, tokens):
        return tokens + self.projection(tokens)


class _DummyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(128, 512)
        self.positional_embedding = nn.Parameter(torch.randn(8, 512))
        self.transformer = SimpleNamespace(
            resblocks=nn.ModuleList(_TextBlock() for _ in range(4))
        )
        self.ln_final = nn.LayerNorm(512)
        self.text_projection = nn.Parameter(torch.eye(512))

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype


class _PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 768, 14, stride=14)

    def forward(self, image):
        return self.projection(image).flatten(2).transpose(1, 2)


class _VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(768, 768)

    def forward(self, tokens, text_features=None):
        return tokens + self.projection(tokens)


class _DummyDino(nn.Module):
    patch_size = 14
    num_register_tokens = 4

    def __init__(self):
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.register_tokens = nn.Parameter(torch.zeros(1, 4, 768))
        self.blocks = nn.ModuleList(_VisionBlock() for _ in range(4))
        self.norm = nn.LayerNorm(768)

    def interpolate_pos_encoding(self, tokens, height, width):
        return torch.zeros_like(tokens)


class ReciprocalAdapterTest(unittest.TestCase):
    def test_initialization_preserves_both_backbone_streams(self):
        torch.manual_seed(0)
        adapter = ReciprocalVisionLanguageAdapter(
            visual_dim=32,
            text_dim=24,
            hidden_dim=16,
            num_heads=4,
            pool_size=2,
        ).eval()
        visual = torch.randn(2, 10, 32)
        text = torch.randn(2, 6, 24)
        padding = torch.zeros(2, 6, dtype=torch.bool)
        visual_out, text_out = adapter(
            visual, text, padding, visual_grid=(1, 5), special_tokens=5
        )
        torch.testing.assert_close(visual_out, visual)
        torch.testing.assert_close(text_out, text)

    def test_padding_values_do_not_change_valid_outputs(self):
        torch.manual_seed(1)
        adapter = ReciprocalVisionLanguageAdapter(
            visual_dim=32,
            text_dim=24,
            hidden_dim=16,
            num_heads=4,
            pool_size=2,
        ).eval()
        nn.init.normal_(adapter.visual_output.weight, std=0.1)
        nn.init.normal_(adapter.text_output.weight, std=0.1)
        visual = torch.randn(2, 11, 32)
        text = torch.randn(2, 6, 24)
        padding = torch.tensor(
            [
                [False, False, False, True, True, True],
                [False, False, True, True, True, True],
            ]
        )
        altered = text.clone()
        altered[padding] = torch.randn_like(altered[padding]) * 1000.0
        visual_a, text_a = adapter(
            visual, text, padding, visual_grid=(2, 3), special_tokens=5
        )
        visual_b, text_b = adapter(
            visual, altered, padding, visual_grid=(2, 3), special_tokens=5
        )
        torch.testing.assert_close(visual_a, visual_b, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            text_a[~padding], text_b[~padding], rtol=1e-5, atol=1e-6
        )
        self.assertTrue(torch.equal(text_a[padding], torch.zeros_like(text_a[padding])))

    def test_zero_initialized_bridge_receives_output_gradients(self):
        torch.manual_seed(3)
        adapter = ReciprocalVisionLanguageAdapter(
            visual_dim=32,
            text_dim=24,
            hidden_dim=16,
            num_heads=4,
            pool_size=2,
        )
        visual = torch.randn(2, 11, 32, requires_grad=True)
        text = torch.randn(2, 6, 24, requires_grad=True)
        padding = torch.zeros(2, 6, dtype=torch.bool)
        visual_out, text_out = adapter(
            visual, text, padding, visual_grid=(2, 3), special_tokens=5
        )
        (visual_out.mean() + text_out.mean()).backward()
        self.assertGreater(adapter.visual_output.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(adapter.text_output.weight.grad.abs().sum().item(), 0.0)

    def test_fusion_runs_layerwise_on_non_square_grid(self):
        torch.manual_seed(2)
        fusion = FUSION.Fusion(
            num_layers=4,
            dino_layers=4,
            output_dinov2=[1, 2],
            fusion_adapter="reciprocal",
            adapter_layers=[0, 2],
            adapter_visual_dim=768,
            adapter_text_dim=512,
            adapter_hidden_dim=64,
            adapter_heads=4,
            adapter_pool_size=2,
            input_is_clip_normalized=True,
        )
        image = torch.randn(2, 3, 28, 42, requires_grad=True)
        text = torch.tensor(
            [
                [1, 5, 127, 0, 0, 0, 0, 0],
                [1, 3, 9, 127, 0, 0, 0, 0],
            ]
        )
        features, tokens, state = fusion(
            image, text, _DummyClip(), _DummyDino()
        )
        self.assertEqual(len(features), 3)
        self.assertTrue(
            all(tuple(value.shape) == (2, 768, 2, 3) for value in features)
        )
        self.assertEqual(tuple(tokens.shape), (2, 8, 512))
        self.assertEqual(tuple(state.shape), (2, 512))
        loss = sum(value.mean() for value in features) + state.mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())


    def test_grasp_aware_fusion_uses_adapter_free_interleaved_path(self):
        torch.manual_seed(6)
        fusion = FUSION.Fusion(
            num_layers=4,
            dino_layers=4,
            output_dinov2=[1, 2],
            fusion_adapter="grasp_aware",
            adapter_layers=[],
            adapter_visual_dim=768,
            adapter_text_dim=512,
            adapter_hidden_dim=64,
            adapter_heads=4,
            adapter_pool_size=2,
            input_is_clip_normalized=True,
        )
        self.assertEqual(len(fusion.reciprocal_adapters), 0)
        image = torch.randn(2, 3, 28, 42)
        text = torch.tensor(
            [
                [1, 5, 127, 0, 0, 0, 0, 0],
                [1, 3, 9, 127, 0, 0, 0, 0],
            ]
        )
        features, tokens, state = fusion(
            image, text, _DummyClip(), _DummyDino()
        )
        self.assertEqual(len(features), 3)
        self.assertTrue(
            all(tuple(value.shape) == (2, 768, 2, 3) for value in features)
        )
        self.assertEqual(tuple(tokens.shape), (2, 8, 512))
        self.assertEqual(tuple(state.shape), (2, 512))
if __name__ == "__main__":
    unittest.main()
