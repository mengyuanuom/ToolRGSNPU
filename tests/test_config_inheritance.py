from pathlib import Path
import tempfile
import unittest

from utils.config import load_cfg_from_cfg_file, merge_cfg_from_list


ROOT = Path(__file__).resolve().parents[1]


class ConfigInheritanceTest(unittest.TestCase):
    def test_etrg_experiment_composes_four_base_configs(self):
        cfg = load_cfg_from_cfg_file(
            ROOT / "configs" / "etrg" / "etrg_r50_ocid_vlg.yaml"
        )
        self.assertEqual(cfg.architecture, "etrg")
        self.assertEqual(cfg.dataset, "OCID-VLG")
        self.assertEqual(cfg.runner["type"], "npu_grasp")
        self.assertEqual(cfg.optim_wrapper["type"], "npu_amp")
        self.assertEqual(cfg.param_scheduler["milestones"], [35])
        self.assertEqual(cfg.epochs, 40)
        self.assertEqual(cfg.sections.MODEL.word_dim, 1024)
        self.assertEqual(cfg.etrg_input_mode, "rgb")
        self.assertFalse(cfg.with_depth)

    def test_cli_override_updates_flat_and_hierarchical_views(self):
        cfg = load_cfg_from_cfg_file(
            ROOT / "configs" / "etrg" / "etrg_r101_ocid_vlg.yaml"
        )
        updated = merge_cfg_from_list(
            cfg, ["TRAIN.batch_size", "1", "DATA.root_path", "/tmp/ocid"]
        )
        self.assertEqual(updated.batch_size, 1)
        self.assertEqual(updated.sections.TRAIN.batch_size, 1)
        self.assertEqual(updated.root_path, "/tmp/ocid")
        self.assertEqual(updated.sections.DATA.root_path, "/tmp/ocid")

    def test_cli_override_supports_deep_component_paths(self):
        cfg = load_cfg_from_cfg_file(
            ROOT / "configs" / "etrg" / "etrg_r50_ocid_vlg.yaml"
        )
        updated = merge_cfg_from_list(
            cfg, ["RUNTIME.param_scheduler.milestones", "[20, 30]"]
        )
        self.assertEqual(updated.param_scheduler["milestones"], [20, 30])
        self.assertEqual(
            updated.sections.RUNTIME.param_scheduler.milestones, [20, 30]
        )

    def test_delete_replaces_inherited_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text(
                "MODEL:\n  neck:\n    type: old\n    width: 64\n",
                encoding="utf-8",
            )
            (root / "child.yaml").write_text(
                "_base_: base.yaml\nMODEL:\n  neck:\n    _delete_: true\n    type: new\n",
                encoding="utf-8",
            )
            cfg = load_cfg_from_cfg_file(root / "child.yaml")
            self.assertEqual(cfg.neck, {"type": "new"})

    def test_non_training_sections_remain_namespaced(self):
        cfg = load_cfg_from_cfg_file(
            ROOT / "config" / "deployment" / "lab.example.yaml"
        )
        self.assertEqual(cfg.camera["type"], "opencv")
        self.assertEqual(cfg.robot["type"], "legacy_tcp")
        self.assertNotIn("type", cfg)


if __name__ == "__main__":
    unittest.main()
