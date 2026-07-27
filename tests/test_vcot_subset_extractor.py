import csv
import io
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from tools.extract_vcot_subset import (
    ConcatenatedParts,
    build_plan,
    load_split_ids,
)


class VCoTSubsetExtractorTest(unittest.TestCase):
    def test_raw_archive_parts_are_exposed_as_one_seekable_zip(self):
        payload = io.BytesIO()
        with ZipFile(payload, "w") as archive:
            archive.writestr("image/scene-a.jpg", b"image-a")
            archive.writestr("image/scene-b.jpg", b"image-b")
        content = payload.getvalue()
        split_at = len(content) // 2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "image_part_aa"
            second = root / "image_part_ab"
            first.write_bytes(content[:split_at])
            second.write_bytes(content[split_at:])

            with ConcatenatedParts([first, second]) as stream:
                with ZipFile(stream) as archive:
                    self.assertEqual(
                        archive.read("image/scene-b.jpg"),
                        b"image-b",
                    )
                    plan = build_plan(
                        archive,
                        {"scene-a", "scene-b"},
                        "image",
                        ".jpg",
                    )
                    self.assertEqual(len(plan), 2)
                    self.assertLessEqual(plan[0].header_offset, plan[1].header_offset)

    def test_split_loader_deduplicates_scenes_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {
                "train.csv": [
                    ["scene-a_0", "cup", "a cup"],
                    ["scene-a_1", "spoon", "a spoon"],
                ],
                "test_seen.csv": [["scene-b_0", "cup", "another cup"]],
            }
            for filename, values in rows.items():
                with (root / filename).open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    csv.writer(stream).writerows(values)

            grasp_ids, scene_ids, files, counts = load_split_ids(
                root, ("train", "seen")
            )
            self.assertEqual(len(grasp_ids), 3)
            self.assertEqual(scene_ids, {"scene-a", "scene-b"})
            self.assertEqual(len(files), 2)
            self.assertEqual(counts, {"train": 2, "seen": 1})


if __name__ == "__main__":
    unittest.main()
