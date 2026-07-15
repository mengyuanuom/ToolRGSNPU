import glob
import os
import unittest

import numpy as np
import torch
import yaml

from model import MODEL_REGISTRY
from model.graspmamba import HierarchicalFeatureFusion
from model.layers import OffsetMultiTaskProjector
from model.lgd import CosineDiffusion, LGDCore
from utils.dataset import GraspTransforms, make_dense_offset_with_radius_np
from utils.data_builder import DATASET_REGISTRY
from utils.ocid_vlg_dataset import parse_ocid_image_filename, resolve_ocid_vlg_split
from utils.vcot_dataset import grasp_anything_to_quads, resolve_vcot_split


class ToolRGSContractsTest(unittest.TestCase):
    def test_dataset_registry_includes_supported_datasets(self):
        self.assertIn("grasptool", DATASET_REGISTRY)
        self.assertIn("vcot", DATASET_REGISTRY)
        self.assertIn("ocid_vlg", DATASET_REGISTRY)
        self.assertEqual(resolve_vcot_split("train"), "train.csv")
        self.assertEqual(resolve_vcot_split("seen"), "test_seen.csv")
        self.assertEqual(resolve_vcot_split("unseen"), "test_unseen.csv")
        self.assertEqual(resolve_ocid_vlg_split("train"), "train_expressions.json")
        self.assertEqual(resolve_ocid_vlg_split("val"), "val_expressions.json")
        self.assertEqual(resolve_ocid_vlg_split("test"), "test_expressions.json")
        self.assertEqual(
            parse_ocid_image_filename("ARID20/floor/seq01,rgb_0001.png"),
            ("ARID20/floor/seq01", "rgb_0001.png"),
        )

    def test_grasp_masks_do_not_wrap_negative_pixels(self):
        transform = GraspTransforms(width_factor=100, width=32, height=32)
        raw = transform.generate_masks(
            np.array([[0.0, 0.0, 30.0, 20.0, 0.0, 0.0]], dtype=np.float32)
        )
        self.assertEqual(raw["pos"][-1, -1], 0)

    def test_vcot_grasp_six_column_conversion(self):
        quads, scores = grasp_anything_to_quads(
            [[0.8, 100.0, 50.0, 40.0, 20.0, 0.0]]
        )
        self.assertEqual(quads.shape, (1, 4, 2))
        np.testing.assert_allclose(quads[0].mean(axis=0), [100.0, 50.0], atol=1e-5)
        np.testing.assert_allclose(
            quads[0], [[80.0, 40.0], [80.0, 60.0], [120.0, 60.0], [120.0, 40.0]]
        )
        edge_lengths = np.linalg.norm(np.roll(quads[0], -1, axis=0) - quads[0], axis=1)
        np.testing.assert_allclose(np.sort(edge_lengths), [20.0, 20.0, 40.0, 40.0])
        np.testing.assert_allclose(scores, [0.8])
        target = GraspTransforms(width_factor=100)(quads, target=0)
        np.testing.assert_allclose(target[0, :5], [100.0, 50.0, 40.0, 20.0, 0.0])

    def test_experiment_configs_reference_registered_models(self):
        expected = {
            "crog",
            "crogoff",
            "drog",
            "drogoff",
            "ggcnnclip",
            "grconvnetclip",
            "graspmamba",
            "lgd",
        }
        self.assertTrue(expected.issubset(MODEL_REGISTRY))
        paths = [
            path
            for directory in ("grasp_tools", "vcot", "ocid_vlg")
            for path in glob.glob(f"config/{directory}/*.yaml")
        ]
        self.assertGreaterEqual(len(paths), 24)
        for path in paths:
            with open(path, encoding="utf-8") as stream:
                cfg = yaml.safe_load(stream)
            architecture = cfg["MODEL"]["architecture"]
            self.assertIn(architecture, MODEL_REGISTRY, path)
            dataset = cfg["DATA"]["dataset"].lower().replace("-", "_")
            self.assertIn(dataset, DATASET_REGISTRY, path)

        for directory, dataset_name in (
            ("grasp_tools", "grasptool"),
            ("vcot", "vcot"),
            ("ocid_vlg", "ocid_vlg"),
        ):
            configs = glob.glob(f"config/{directory}/*.yaml")
            self.assertEqual(
                {os.path.splitext(os.path.basename(path))[0] for path in configs},
                expected,
                directory,
            )
            for path in configs:
                with open(path, encoding="utf-8") as stream:
                    cfg = yaml.safe_load(stream)
                actual = cfg["DATA"]["dataset"].lower().replace("-", "_")
                self.assertEqual(actual, dataset_name, path)

    def test_offset_projector_output_contract(self):
        projector = OffsetMultiTaskProjector(word_dim=512, in_dim=256)
        features = torch.randn(2, 512, 8, 8)
        text_state = torch.randn(2, 512)
        outputs = projector(features, text_state)
        self.assertEqual(len(outputs), 6)
        for output in outputs[:5]:
            self.assertEqual(tuple(output.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(outputs[5].shape), (2, 2, 32, 32))
        self.assertLessEqual(outputs[5].abs().max().item(), 1.0)

    def test_dense_offset_points_toward_grasp_center(self):
        center = np.array([[8.0, 9.0]], dtype=np.float32)
        offset, weight = make_dense_offset_with_radius_np(
            centers_xy=center,
            img_size_hw=(20, 20),
            r_pix=4.0,
            use_gaussian=True,
        )
        self.assertEqual(offset.shape, (2, 20, 20))
        self.assertEqual(weight.shape, (1, 20, 20))
        np.testing.assert_allclose(offset[:, 9, 8], 0.0, atol=1e-6)
        self.assertGreater(offset[0, 9, 6], 0.0)
        self.assertGreater(weight[0, 9, 8], weight[0, 9, 6])
        self.assertLessEqual(np.abs(offset).max(), 1.0)

    def test_lgd_dense_diffusion_contract(self):
        diffusion = CosineDiffusion(timesteps=20)
        clean = torch.zeros(2, 1, 64, 64)
        noisy = diffusion.q_sample(clean, torch.tensor([0, 19]))
        self.assertEqual(tuple(noisy.shape), tuple(clean.shape))
        self.assertTrue(torch.all(diffusion.betas >= 0))
        self.assertTrue(torch.all(diffusion.betas < 1))

        core = LGDCore(word_dim=32, base_channels=16, time_dim=32)
        outputs = core(
            torch.randn(2, 3, 64, 64),
            noisy,
            torch.tensor([0, 19]),
            torch.randn(2, 32),
        )
        self.assertEqual(len(outputs), 5)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (2, 1, 64, 64))

    def test_graspmamba_hierarchical_fusion_contract(self):
        fusion = HierarchicalFeatureFusion(
            visual_channels=(8, 16, 32, 64),
            text_dim=24,
            fusion_dim=12,
        )
        features = (
            torch.randn(2, 8, 32, 32),
            torch.randn(2, 16, 16, 16),
            torch.randn(2, 32, 8, 8),
            torch.randn(2, 64, 4, 4),
        )
        output = fusion(features, torch.randn(2, 24))
        self.assertEqual(tuple(output.shape), (2, 12, 32, 32))


if __name__ == "__main__":
    unittest.main()
