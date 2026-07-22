from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from toolrgs.engine.runner import NPUGraspRunner


class _Stateful:
    def __init__(self, name):
        self.name = name

    def state_dict(self):
        return {"name": self.name}


class CheckpointPolicyTest(unittest.TestCase):
    def test_last_and_each_validation_best_are_tracked_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = NPUGraspRunner.__new__(NPUGraspRunner)
            runner.is_main = True
            runner.cfg = SimpleNamespace(
                output_dir=directory,
                architecture="crog",
                filename="config/grasp_tools/crog.yaml",
            )
            runner.model = _Stateful("model")
            runner.optimizer = _Stateful("optimizer")
            runner.scheduler = _Stateful("scheduler")
            runner.best_iou = 0.0
            runner.best_j1 = 0.0
            runner.best_j5 = 0.0
            payloads = []

            def fake_save(payload, path):
                payloads.append(payload)
                Path(path).write_text(str(payload["epoch"]), encoding="utf-8")

            with patch("toolrgs.engine.runner.torch.save", side_effect=fake_save):
                runner.save_checkpoint(
                    1,
                    {
                        "validation": {
                            "iou": 0.50,
                            "precision": {},
                            "j_index": [0.40, 0.60],
                        }
                    },
                )
                runner.save_checkpoint(
                    2,
                    {
                        "validation": {
                            "iou": 0.45,
                            "precision": {},
                            "j_index": [0.35, 0.70],
                        }
                    },
                )
                runner.save_checkpoint(
                    3,
                    {
                        "validation": {
                            "iou": 0.44,
                            "precision": {},
                            "j_index": [0.34, 0.65],
                        }
                    },
                )

            root = Path(directory)
            self.assertEqual((root / "last.pth").read_text(encoding="utf-8"), "3")
            self.assertEqual(
                (root / "best_iou_epoch_001.pth").read_text(encoding="utf-8"),
                "1",
            )
            self.assertEqual(
                (root / "best_j1_epoch_001.pth").read_text(encoding="utf-8"),
                "1",
            )
            self.assertFalse((root / "best_j5_epoch_001.pth").exists())
            self.assertEqual(
                (root / "best_j5_epoch_002.pth").read_text(encoding="utf-8"),
                "2",
            )
            self.assertEqual(len(payloads), 3)
            self.assertEqual(payloads[-1]["best_iou"], 0.50)
            self.assertEqual(payloads[-1]["best_j1_index"], 0.40)
            self.assertEqual(payloads[-1]["best_j5_index"], 0.70)
            self.assertEqual(
                payloads[-1]["validation"],
                {"iou": 0.44, "j_at_one": 0.34, "j_at_five": 0.65},
            )


if __name__ == "__main__":
    unittest.main()
