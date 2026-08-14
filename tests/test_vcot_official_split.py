from pathlib import Path
import tempfile
import unittest

try:
    from utils.vcot_dataset import VCoTDataset
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    VCoTDataset = None


ROOT = Path(__file__).resolve().parents[1]


class VCoTOfficialSplitTest(unittest.TestCase):
    def _make_dataset_root(self, directory):
        root = Path(directory) / "data"
        split_root = Path(directory) / "split"
        (root / "grasp_label_positive").mkdir(parents=True)
        split_root.mkdir()
        rows = [f"scene_{index},object,description\n" for index in range(7)]
        (split_root / "train.csv").write_text("".join(rows), encoding="utf-8")
        return root, split_root

    @unittest.skipIf(VCoTDataset is None, "PyTorch is not installed")
    def test_official_train_and_validation_partition_train_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root, split_root = self._make_dataset_root(directory)
            train = VCoTDataset(
                root,
                split="train_official",
                split_root=split_root,
                vcot_official_val_size=2,
            )
            validation = VCoTDataset(
                root,
                split="val_official",
                split_root=split_root,
                vcot_official_val_size=2,
            )
            self.assertEqual(len(train), 5)
            self.assertEqual(len(validation), 2)
            self.assertEqual(train.samples[-1][0], "scene_4")
            self.assertEqual(validation.samples[0][0], "scene_5")

    def test_evaluate_accepts_explicit_seen_or_unseen_override(self):
        source = (ROOT / "evaluate.py").read_text(encoding="utf-8-sig")
        self.assertIn('parser.add_argument(\n        "--split"', source)
        self.assertIn("build_dataset(args, args.eval_split", source)


if __name__ == "__main__":
    unittest.main()
