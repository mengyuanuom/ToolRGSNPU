import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V3_DIR = ROOT / "assets" / "grasp_tools" / "graspall_v3"


def test_v3_source_release_is_complete_and_valid():
    images = sorted(V3_DIR.glob("*.jpg"))
    annotations = sorted(V3_DIR.glob("*.json"))

    assert len(images) == 107
    assert len(annotations) == 107
    assert {path.stem for path in images} == {path.stem for path in annotations}

    categories = set()
    object_count = 0
    grasp_count = 0
    for annotation_path in annotations:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        objects = payload.get("objects")
        assert isinstance(objects, list) and objects
        for obj in objects:
            object_count += 1
            categories.add(obj["category"])
            mask = obj.get("mask")
            grasps = obj.get("grasps")
            assert isinstance(mask, list) and len(mask) >= 3
            assert isinstance(grasps, list) and grasps
            assert len(obj.get("bbox", [])) == 4
            assert all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for point in mask
                for value in point
            )
            for grasp in grasps:
                assert len(grasp) == 4
                assert all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for point in grasp
                    for value in point
                )
            grasp_count += len(grasps)

    assert object_count == 107
    assert len(categories) == 22
    assert grasp_count == 7820
