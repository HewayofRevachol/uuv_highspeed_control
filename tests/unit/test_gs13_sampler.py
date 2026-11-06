"""Real-plant scenario coverage and reproducibility contracts for GS-13."""
from collections import Counter
from itertools import product
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments.gs12_gate import plant_harness
from experiments.gs13_srppo import sampler


def _tuple(case):
    return case["speed"], case["family"], case["maneuver"]


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_full_cycles_cover_cartesian_matrix_and_small_pilots_cover_maneuvers(split):
    cases = sampler.build_cases(split, 2 * sampler.CYCLE_SIZE, 41002, duration_s=2.)
    expected = set(product(sampler.SPEEDS, sampler.FAMILIES, sampler.MANEUVERS))
    assert set(map(_tuple, cases[:sampler.CYCLE_SIZE])) == expected
    assert set(map(_tuple, cases[sampler.CYCLE_SIZE:])) == expected
    assert set(c["maneuver"] for c in cases[:5]) == set(sampler.MANEUVERS)
    assert set(c["family"] for c in cases[:10]) == set(sampler.FAMILIES)
    assert Counter(c["speed"] for c in cases[:3]) == Counter(sampler.SPEEDS)
    assert all(c["speed"] == c["requested_speed"] for c in cases)
    assert all(c["duration_s"] == 2. and "plant" in c for c in cases)
    assert all(np.isfinite(plant_harness.Reference(c).at(.2)).all() for c in cases)


def test_cases_are_repeatable_prefix_stable_and_split_seeds_disjoint():
    cases = {split: sampler.build_cases(split, 150, 41002)
             for split in ("train", "dev", "test")}
    for split, rows in cases.items():
        assert rows == sampler.build_cases(split, 150, 41002)
        assert rows[:10] == sampler.build_cases(split, 10, 41002)
        start = sampler.SPLIT_SEED_START[split]
        assert all(start <= c["seed"] < start + sampler.SPLIT_SEED_WIDTH for c in rows)
        assert all(start <= c["noise_seed"] < start + sampler.SPLIT_SEED_WIDTH for c in rows)
        assert len({c["seed"] for c in rows}) == len(rows)
    for field in ("case_id", "seed", "noise_seed"):
        all_values = [c[field] for rows in cases.values() for c in rows]
        assert len(all_values) == len(set(all_values))
    dev = {_tuple(c): c for c in cases["dev"]}
    test = {_tuple(c): c for c in cases["test"]}
    assert all(dev[key]["phase"] != test[key]["phase"] for key in dev)


def test_sampler_is_independent_of_python_hash_seed():
    command = [sys.executable, "-c", (
        "import json; from experiments.gs13_srppo.sampler import build_cases; "
        "print(json.dumps(build_cases('train', 5, 41002), sort_keys=True))")]
    first = subprocess.check_output(command, env={**os.environ, "PYTHONHASHSEED": "7"}, text=True)
    second = subprocess.check_output(command, env={**os.environ, "PYTHONHASHSEED": "999"}, text=True)
    assert json.loads(first) == json.loads(second)


def test_capacity_filter_uses_real_trim_and_preserves_evaluation_speeds():
    train = sampler.build_cases("train", 300, 41002)
    assert all(not c["capacity_limited"] for c in train)
    assert all(not plant_harness.capacity_limited(c) for c in train)
    assert sum(c["sampling"]["capacity_rejections"] for c in train) > 0
    for split in ("dev", "test"):
        cases = sampler.build_cases(split, sampler.CYCLE_SIZE, 41002)
        assert all(c["capacity_limited"] == plant_harness.capacity_limited(c) for c in cases)
        for family in ("damping", "thrust"):
            relevant = [c for c in cases if c["speed"] == 18. and c["family"] == family]
            assert len(relevant) == len(sampler.MANEUVERS)
            assert all(c["capacity_limited"] for c in relevant)
        assert all(c["speed"] == c["requested_speed"] for c in cases)


def test_combination_capacity_is_derived_from_plant_not_family_label(monkeypatch):
    original = plant_harness.make_case

    def heavy_combination(*args, **kwargs):
        case = original(*args, **kwargs)
        if case["family"] == "combination":
            case["plant"].update(damping=1.1, thrust=.9)
            case["capacity_limited"] = False  # Deliberately stale input metadata.
        return case

    monkeypatch.setattr(plant_harness, "make_case", heavy_combination)
    cases = sampler.build_cases("dev", sampler.CYCLE_SIZE, 41002)
    heavy = [c for c in cases if c["family"] == "combination" and c["speed"] == 18.]
    assert len(heavy) == len(sampler.MANEUVERS)
    assert all(c["capacity_limited"] for c in heavy)


def test_infeasible_training_retries_are_bounded_and_do_not_change_tuple(monkeypatch):
    visited = []
    original = plant_harness.make_case

    def record(*args, **kwargs):
        case = original(*args, **kwargs)
        visited.append(case)
        return case

    monkeypatch.setattr(plant_harness, "make_case", record)
    monkeypatch.setattr(plant_harness, "capacity_limited", lambda _: True)
    with pytest.raises(RuntimeError, match="speed was not changed"):
        sampler.build_cases("train", 1, 41002)
    assert len(visited) == sampler.MAX_CAPACITY_ATTEMPTS
    assert len({_tuple(c) for c in visited}) == 1
    assert len({c["seed"] for c in visited}) == sampler.MAX_CAPACITY_ATTEMPTS


def test_compatibility_wrapper_exposes_real_cases_and_validates_requests():
    wrapper = sampler.StratifiedSampler(41002)
    train, dev, test = [wrapper.generate(split, 5) for split in ("train", "dev", "test")]
    wrapper.assert_disjoint(train, dev, test)
    assert "plant" in train[0].as_dict()
    assert [len(batch) for batch in wrapper.iter_batches("train", 5, 2)] == [2, 2, 1]
    assert sampler.build_cases("train", 0, 41002) == []
    with pytest.raises(ValueError, match="split"):
        sampler.build_cases("unknown", 1, 41002)
    with pytest.raises(ValueError, match="positive"):
        sampler.build_cases("train", 1, 41002, duration_s=float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        sampler.build_cases("train", -1, 41002)
