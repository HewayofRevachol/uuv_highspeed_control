"""Reproducible, stratified real-plant cases for GS-13.

Each cycle covers the Cartesian product of speeds, plant families and
maneuvers. Training explicitly rejects capacity-limited plant draws without
changing their requested speed; development and test retain those draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Any, Iterator

import numpy as np

from experiments.gs12_gate import plant_harness


SPEEDS = (12., 15., 18.)
FAMILIES = ("nominal", "fin", "mass", "damping", "thrust", "delay", "tau",
            "second_all", "second_fins", "combination")
MANEUVERS = ("straight", "speed", "turn", "depth", "coupled")
SPLIT_SEED_START = {"train": 100_000_000, "dev": 400_000_000,
                    "test": 700_000_000}
SPLIT_SEED_WIDTH = 100_000_000
MAX_CAPACITY_ATTEMPTS = 32
CYCLE_SIZE = len(SPEEDS) * len(FAMILIES) * len(MANEUVERS)


def _cycles(rng: np.random.Generator) -> Iterator[tuple[int, float, str, str]]:
    """Shuffle factor orders while preserving balanced prefixes and cycles.

    The extra maneuver rotation after every 30 entries makes every
    speed/family/maneuver tuple appear once per 150 entries. A five-entry
    prefix visits every maneuver; a ten-entry prefix visits every family.
    """
    cycle = 0
    while True:
        speeds = rng.permutation(SPEEDS)
        families = rng.permutation(FAMILIES)
        maneuvers = rng.permutation(MANEUVERS)
        for index in range(CYCLE_SIZE):
            yield (cycle, float(speeds[index % 3]), str(families[index % 10]),
                   str(maneuvers[(index % 5 + index // 30) % 5]))
        cycle += 1


def build_cases(split: str, count: int, seed: int,
                duration_s: float = 24.) -> list[dict[str, Any]]:
    """Build real GS-12 cases with disjoint split seed namespaces.

    ``train`` contains only plants with feasible nominal straight-line trim;
    this does not imply every maneuver is feasible. Rejected physical draws
    are counted in the accepted case's sampling metadata. After 32 rejected
    draws for one tuple, fail explicitly instead of relabeling its speed or
    family. ``dev`` and ``test`` retain capacity-limited cases.
    """
    if split not in SPLIT_SEED_START:
        raise ValueError("split must be train, dev or test")
    if isinstance(count, bool) or isinstance(seed, bool):
        raise ValueError("count and seed must be integers")
    count, seed = operator.index(count), operator.index(seed)
    duration_s = float(duration_s)
    if count < 0 or seed < 0 or not np.isfinite(duration_s) or duration_s <= 0.:
        raise ValueError("count/seed must be non-negative and duration_s positive")
    # GS-12 derives noise_seed = seed + 10,000. Keep these derived seeds in
    # the same split namespace too.
    available_seeds = SPLIT_SEED_WIDTH - 10_000
    if count * MAX_CAPACITY_ATTEMPTS > available_seeds:
        raise ValueError("requested case count exceeds the split seed budget")
    if count == 0:
        return []

    split_code = tuple(SPLIT_SEED_START).index(split)
    layout_rng = np.random.default_rng(np.random.SeedSequence([seed, split_code, 0]))
    seed_rng = np.random.default_rng(np.random.SeedSequence([seed, split_code, 1]))
    # A coprime affine sequence gives unique candidate seeds and stable
    # prefixes regardless of the requested count, without duplicate retries.
    offset = int(seed_rng.integers(available_seeds))
    stride = 10_007  # Coprime to available_seeds=99,990,000.
    candidate_index = 0
    cases: list[dict[str, Any]] = []
    for cycle, speed, family, maneuver in _cycles(layout_rng):
        case_id = f"{split}_{len(cases):05d}"
        rejected = 0
        for _ in range(MAX_CAPACITY_ATTEMPTS):
            case_seed = SPLIT_SEED_START[split] + (offset + stride * candidate_index) % available_seeds
            candidate_index += 1
            case = plant_harness.make_case(case_id, split, speed, family, maneuver,
                                           case_seed, duration_s=duration_s,
                                           training=split == "train")
            limited = plant_harness.capacity_limited(case)
            case["capacity_limited"] = limited
            if split == "train" and limited:
                rejected += 1
                continue
            # Independent evaluation references avoid identical phase-zero
            # dev/test trajectories despite distinct seed identifiers.
            if split != "train":
                reference_rng = np.random.default_rng(np.random.SeedSequence([case_seed, 2]))
                case["amplitude"] = float(reference_rng.uniform(.8, 1.1))
                case["omega"] = float(.5 * reference_rng.uniform(.9, 1.1))
                case["phase"] = float(reference_rng.uniform(0., 2. * np.pi))
                case["initial_offset"] = reference_rng.uniform(-.05, .05, 3).tolist()
            case["requested_speed"] = speed
            case["sampling"] = {
                "scheme": "gs13_cartesian_cycles_v1", "seed": seed,
                "cycle": cycle, "candidate_index": candidate_index - 1,
                "capacity_rejections": rejected,
                "feasible_training_only": split == "train",
                "split_seed_range": [SPLIT_SEED_START[split],
                                     SPLIT_SEED_START[split] + SPLIT_SEED_WIDTH],
            }
            cases.append(case)
            break
        else:
            raise RuntimeError(
                f"No capacity-feasible training plant after {MAX_CAPACITY_ATTEMPTS} "
                f"attempts for {case_id}: speed={speed}, family={family}, "
                f"maneuver={maneuver}; speed was not changed")
        if len(cases) == count:
            return cases
    raise AssertionError("unreachable: scene cycle iterator is infinite")


@dataclass(frozen=True)
class Scene:
    """Compatibility view; ``as_dict`` retains the complete real plant case."""

    scene_id: str
    split: str
    speed: float
    family: str
    maneuver: str
    seed: int
    capacity_limited: bool
    requested_speed: float | None = None
    case: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {**self.case, "scene_id": self.scene_id, "split": self.split,
                "speed": self.speed, "family": self.family,
                "maneuver": self.maneuver, "seed": self.seed,
                "capacity_limited": self.capacity_limited,
                "requested_speed": self.requested_speed if self.requested_speed is not None else self.speed}


class StratifiedSampler:
    """Compatibility wrapper around :func:`build_cases`.

    ``capacity_fraction`` is accepted for old callers, but never changes speed
    or drops evaluation cases. Training always applies the explicit feasible
    filter; development/test always retain the complete requested matrix.
    """

    def __init__(self, seed: int = 41002, capacity_fraction: float | None = None):
        if capacity_fraction is not None and not 0. < capacity_fraction <= 1.:
            raise ValueError("capacity_fraction must be in (0, 1]")
        self.seed = int(seed)

    def generate(self, split: str, count: int) -> list[Scene]:
        return [Scene(case["case_id"], split, case["speed"], case["family"],
                      case["maneuver"], case["seed"], case["capacity_limited"],
                      case["requested_speed"], case)
                for case in build_cases(split, count, self.seed)]

    def iter_batches(self, split: str, count: int, batch_size: int) -> Iterator[list[Scene]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        scenes = self.generate(split, count)
        for start in range(0, len(scenes), batch_size):
            yield scenes[start:start + batch_size]

    @staticmethod
    def assert_disjoint(*scene_lists: list[Scene]) -> None:
        scenes = [scene for group in scene_lists for scene in group]
        if len({scene.scene_id for scene in scenes}) != len(scenes):
            raise ValueError("scene id leaked across train/dev/test split")
        if len({scene.seed for scene in scenes}) != len(scenes):
            raise ValueError("scene seed leaked across train/dev/test split")
