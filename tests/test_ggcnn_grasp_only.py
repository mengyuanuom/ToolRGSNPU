import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
import torch.nn as nn
import yaml

from toolrgs.models.base import model_predicts_segmentation
from toolrgs.structures import GraspModelResult, GraspOutput


ROOT = Path(__file__).resolve().parents[1]


def _load_ggcnn_class():
    package_name = "_ggcnn_grasp_only_test_model"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "model")]
    sys.modules[package_name] = package
    for module_name in ("crog_clip", "ggcnnclip"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name, ROOT / "model" / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.ggcnnclip"].GGCNN_CLIP


GGCNN_CLIP = _load_ggcnn_class()


class _TextBackbone(nn.Module):
    def encode_text(self, word):
        state = torch.ones(word.shape[0], 4, device=word.device)
        return None, state


class _DenseGraspHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image, _state):
        base = image[:, :1] * self.scale
        return base, 2.0 * base, 3.0 * base, 4.0 * base


class GGCNNGraspOnlyTest(unittest.TestCase):
    @staticmethod
    def _model():
        model = GGCNN_CLIP.__new__(GGCNN_CLIP)
        nn.Module.__init__(model)
        model.backbone = _TextBackbone()
        model.grasp_head = _DenseGraspHead()
        model.predicts_grasp_short_side = False
        model.short_side_loss_weight = 1.0
        return model

    def test_structured_output_allows_no_segmentation(self):
        output = GraspOutput(
            segmentation=None,
            quality=torch.zeros(1, 1, 2, 2),
            sine=torch.zeros(1, 1, 2, 2),
            cosine=torch.ones(1, 1, 2, 2),
            width=torch.ones(1, 1, 2, 2),
        )
        self.assertIsNone(output.segmentation)
        self.assertIsNone(output.detach().segmentation)

    def test_capability_contract_marks_ggcnn_as_grasp_only(self):
        self.assertFalse(model_predicts_segmentation(self._model()))

        class LegacyModel:
            pass

        self.assertTrue(model_predicts_segmentation(LegacyModel()))

    def test_forward_returns_only_grasp_maps_and_raw_mse_loss(self):
        model = self._model().train()
        image = torch.ones(2, 3, 8, 8)
        word = torch.zeros(2, 4, dtype=torch.long)
        target = torch.zeros(2, 1, 4, 4)
        result = model(
            image,
            word,
            torch.ones_like(target),
            target,
            target,
            target,
            target,
        )

        self.assertIsInstance(result, GraspModelResult)
        self.assertIsNone(result.predictions.segmentation)
        self.assertIsNone(result.targets.segmentation)
        self.assertAlmostEqual(result.loss.item(), 30.0, places=5)
        result.loss.backward()
        self.assertIsNotNone(model.grasp_head.scale.grad)

    def test_eval_keeps_quality_as_quality_instead_of_fake_mask(self):
        model = self._model().eval()
        result = model(
            torch.ones(1, 3, 8, 8),
            torch.zeros(1, 4, dtype=torch.long),
        )
        self.assertIsNone(result.predictions.segmentation)
        self.assertEqual(tuple(result.predictions.quality.shape), (1, 1, 8, 8))


    def test_realvlg_profile_uses_new_grasp_only_experiment(self):
        with (ROOT / "config" / "realvlg" / "ggcnnclip.yaml").open(
            encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)
        self.assertIn("grasponly", config["TRAIN"]["exp_name"])
        self.assertEqual(config["TEST"]["grasp_quality_activation"], "identity")
        self.assertEqual(config["TEST"]["grasp_size_activation"], "clamp")
        self.assertFalse(config["TEST"]["evaluate_segmentation"])


if __name__ == "__main__":
    unittest.main()