import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "toolrgs_native_adapter_fusion", ROOT / "model" / "native_adapter.py"
)
NATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE)


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


class _VisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(768, 768)

    def forward(self, tokens):
        return tokens + self.projection(tokens)


class _DummyDino(nn.Module):
    patch_size = 14
    num_register_tokens = 4

    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 768, 14, stride=14)
        self.cls = nn.Parameter(torch.zeros(1, 1, 768))
        self.registers = nn.Parameter(torch.zeros(1, 4, 768))
        self.blocks = nn.ModuleList(_VisionBlock() for _ in range(4))
        self.norm = nn.LayerNorm(768)

    def prepare_tokens_with_masks(self, image, masks=None):
        patches = self.patch(image).flatten(2).transpose(1, 2)
        return torch.cat(
            [
                self.cls.expand(image.shape[0], -1, -1),
                self.registers.expand(image.shape[0], -1, -1),
                patches,
            ],
            dim=1,
        )


class NativeFusionTest(unittest.TestCase):
    def test_layerwise_fusion_shapes_and_alignment(self):
        cfg = SimpleNamespace(
            native_visual_layers=[0, 1, 2, 3],
            native_text_layers=[0, 1, 2, 3],
            native_input_is_clip_normalized=True,
            native_text_adapter_dim=32,
            native_cross_dim=64,
            native_cross_heads=4,
            native_adapter_dropout=0.0,
            native_alignment_dim=32,
        )
        fusion = NATIVE.NativeDinoClipFusion(cfg).eval()
        image = torch.randn(2, 3, 28, 28)
        text = torch.tensor(
            [[1, 5, 8, 127, 0, 0, 0, 0], [1, 3, 127, 0, 0, 0, 0, 0]]
        )
        features, tokens, state, alignment = fusion(
            image, text, _DummyClip(), _DummyDino()
        )
        self.assertEqual(len(features), 4)
        self.assertTrue(all(tuple(value.shape) == (2, 768, 2, 2) for value in features))
        self.assertEqual(tuple(tokens.shape), (2, 8, 512))
        self.assertEqual(tuple(state.shape), (2, 512))
        self.assertEqual(tuple(alignment.shape), (2, 1, 2, 2))
        self.assertTrue(torch.isfinite(alignment).all())


if __name__ == "__main__":
    unittest.main()
