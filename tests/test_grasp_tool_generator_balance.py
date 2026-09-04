from collections import Counter
from pathlib import Path
import random
import sys

from tools.dataset_converters.grasp_tools.augment import (
    BalancedCategorySampler,
    BalancedTransformSampler,
    SourceObject,
    build_config,
    parse_args,
)


def make_source(source_id: str) -> SourceObject:
    return SourceObject(
        source_id=source_id,
        image_path=Path(f"{source_id}.jpg"),
        object_index=0,
        category_key="wrench",
        category_name="wrench",
        mask=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        grasps=(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),),
    )


def test_default_cli_matches_toolrgs_v3_15k_contract(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["augment.py"])
    args = parse_args()

    assert args.src_dir == "assets/grasp_tools/graspall_v3"
    assert args.out_dir == "datasets/grasp-tools/aug_graspall_v3_15k"
    assert (args.train_scenes, args.val_scenes, args.test_scenes) == (
        12000,
        1000,
        2000,
    )
    assert (args.objects_min, args.objects_max) == (2, 3)
    assert (args.queries_min, args.queries_max) == (2, 4)
    assert args.max_query_difficulty == 1
    assert args.language_templates == "shared"
    assert args.category_vocabulary == "expanded"
    assert args.scales == (0.9, 1.0, 1.15, 1.3)
    assert args.angle_bins == 24
    assert args.seed == 2025


def test_smoke_profile_matches_current_toolrgs_generator(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["augment.py", "--smoke-test"])
    config = build_config(parse_args())
    assert (config.train_scenes, config.val_scenes, config.test_scenes) == (4, 2, 2)


def test_category_sampler_balances_each_complete_cycle():
    categories = ["box", "pliers", "wrench", "screwdriver"]
    sampler = BalancedCategorySampler(categories, random.Random(2025))
    first_cycle = [sampler.next() for _ in categories]
    second_cycle = [sampler.next() for _ in categories]
    assert sorted(first_cycle) == sorted(categories)
    assert sorted(second_cycle) == sorted(categories)


def test_transform_sampler_exhausts_source_scale_angle_product():
    sources = [make_source("wrench-a"), make_source("wrench-b")]
    sampler = BalancedTransformSampler(
        {"wrench": sources},
        scales=(0.8, 1.2),
        angle_bins=4,
        rng=random.Random(7),
    )
    samples = [sampler.next("wrench") for _ in range(16)]
    source_counts = Counter(source.source_id for source, _, _ in samples)
    scale_counts = Counter(scale for _, scale, _ in samples)
    assert source_counts == {"wrench-a": 8, "wrench-b": 8}
    assert scale_counts == {0.8: 8, 1.2: 8}
    assert all(0.0 <= angle < 360.0 for _, _, angle in samples)
