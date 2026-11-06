"""Deterministic teacher labels from a small safe gate candidate grid."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .safety import EMERGENCY_GATE

DEFAULT_CANDIDATES = (
    (0., 0., 0., 0.), (1., 1., 1., 1.), (.5, .5, .5, .5),
    (1., 0., 0., 0.), (0., 1., 1., 1.), (1., 1., 0., 1.),
    (1., 0., 1., 1.), (1., 1., 1., 0.), (0., .5, .5, .5),
)


@dataclass(frozen=True)
class TeacherLabel:
    teacher_gate: tuple[float, float, float, float]
    emergency: bool
    feasible: bool
    track_cost: float
    barrier_cost: float
    risk: float
    label_source: str
    candidate_scores: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _gate(value: Any) -> np.ndarray:
    gate = np.asarray(value, dtype=float)
    if gate.shape != (4,) or not np.isfinite(gate).all():
        raise ValueError("teacher candidate gate must be finite [4]")
    return np.clip(gate, 0., 1.)


def build_teacher_label(observation: Any, candidates: Iterable[Any] = DEFAULT_CANDIDATES,
                        limits: Mapping[str, Any] | None = None,
                        scorer: Callable[[Any, np.ndarray], Mapping[str, Any]] | None = None,
                        split: str = "train") -> TeacherLabel:
    """Select the lowest tracking cost among barrier-free candidates.

    Labels are intentionally restricted to training scenes; a dev/test label
    would leak the evaluation protocol.  ``scorer`` adapts a GS-12 episode
    evaluator and is never called by policy code.
    """
    if split != "train":
        raise ValueError("teacher labels may only be generated for train scenes")
    limits = {} if limits is None else limits
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, Mapping):
            gate = _gate(candidate.get("gate", candidate.get("teacher_gate")))
            score = dict(candidate)
            score.setdefault("track_cost", 0.)
            score.setdefault("barrier_cost", 0.)
            score.setdefault("risk", 0.)
            score.setdefault("feasible", True)
        else:
            gate = _gate(candidate)
            score = {}
        if scorer is not None:
            score.update(dict(scorer(observation, gate)))
        for name in ("track_cost", "barrier_cost", "risk"):
            score[name] = float(score.get(name, 0.))
            if not np.isfinite(score[name]) or score[name] < 0.:
                raise ValueError(f"candidate {name} must be finite and non-negative")
        feasible = bool(score.get("feasible", score["barrier_cost"] <= float(limits.get("barrier", 0.))))
        rows.append({"index": index, "gate": gate.tolist(), "track_cost": score["track_cost"],
                     "barrier_cost": score["barrier_cost"], "risk": score["risk"],
                     "feasible": feasible, "source": score.get("source", "candidate")})
    if not rows:
        raise ValueError("candidate grid must not be empty")
    feasible_rows = [row for row in rows if row["feasible"]]
    if not feasible_rows:
        return TeacherLabel(tuple(EMERGENCY_GATE.tolist()), True, False,
                            float(min(row["track_cost"] for row in rows)),
                            float(min(row["barrier_cost"] for row in rows)),
                            float(min(row["risk"] for row in rows)), "emergency",
                            tuple(rows))
    selected = min(feasible_rows, key=lambda row: (row["track_cost"], row["barrier_cost"], row["index"]))
    return TeacherLabel(tuple(selected["gate"]), False, True,
                        float(selected["track_cost"]), float(selected["barrier_cost"]),
                        float(selected["risk"]), str(selected["source"]), tuple(rows))


def fixed_teacher_candidates() -> tuple[tuple[float, float, float, float], ...]:
    return tuple(DEFAULT_CANDIDATES)
