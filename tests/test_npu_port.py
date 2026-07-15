"""Static contracts that prevent accidental CUDA regressions in ToolRGSNPU."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import yaml

from toolrgs.preflight import validate_required_artifacts


ROOT = Path(__file__).resolve().parents[1]


class NPUConfigurationTest(unittest.TestCase):
    def test_missing_weight_error_reports_key_and_resolved_path(self):
        missing = Path(tempfile.gettempdir()) / "toolrgsnpu-missing-clip.pt"
        cfg = SimpleNamespace(clip_pretrain=str(missing))
        with self.assertRaises(FileNotFoundError) as context:
            validate_required_artifacts(cfg)
        message = str(context.exception)
        self.assertIn("clip_pretrain", message)
        self.assertIn(str(missing.resolve()), message)
        self.assertIn("Current working directory", message)

    def test_all_experiment_configs_use_hccl_and_expose_npu_runtime_options(self):
        experiment_roots = ("grasp_tools", "vcot", "ocid_vlg")
        paths = [
            path
            for directory in experiment_roots
            for path in (ROOT / "config" / directory).glob("*.yaml")
        ]
        self.assertEqual(len(paths), 24)
        for path in paths:
            config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            with self.subTest(config=path.name, dataset=path.parent.name):
                self.assertEqual(config["Distributed"]["dist_backend"], "hccl")
                self.assertIn("amp", config["TRAIN"])
                self.assertIn(config["TRAIN"]["optimizer"], ("adam", "npu_fused_adam"))
                self.assertFalse(config["TRAIN"]["pin_memory"])

    def test_deployment_defaults_to_npu(self):
        config = yaml.safe_load(
            (ROOT / "config" / "deployment" / "lab.example.yaml").read_text(
                encoding="utf-8-sig"
            )
        )
        self.assertEqual(config["model"]["device"], "npu:0")
        self.assertEqual(config["detector"]["device"], "npu:0")
        self.assertEqual(config["audio"]["device"], "cpu")


class NPUSourceContractTest(unittest.TestCase):
    def test_accelerated_entrypoints_have_no_cuda_or_nccl_calls(self):
        paths = (
            ROOT / "train.py",
            ROOT / "evaluate.py",
            ROOT / "engine" / "engine.py",
            ROOT / "toolrgs" / "engine" / "loops.py",
            ROOT / "toolrgs" / "engine" / "val_loop.py",
            ROOT / "deployment" / "inference.py",
        )
        forbidden = (".cuda(", "torch.cuda", '"nccl"', "'nccl'")
        for path in paths:
            source = path.read_text(encoding="utf-8-sig")
            with self.subTest(path=path.name):
                for value in forbidden:
                    self.assertNotIn(value, source)

    def test_runtime_is_explicit_and_does_not_monkey_patch_cuda(self):
        source = (ROOT / "toolrgs" / "runtime" / "device.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn("import torch_npu", source)
        self.assertIn("torch_npu.npu.set_device", source)
        self.assertIn("adapter.npu.amp.autocast", source)
        self.assertNotIn("from torch_npu.contrib.transfer_to_npu", source)


if __name__ == "__main__":
    unittest.main()
