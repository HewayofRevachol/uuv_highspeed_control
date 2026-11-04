"""Frozen H15 run.json contract for paired methods and diagnostics."""

import json
import math
from pathlib import Path


SCHEMA = 1
TOP_LEVEL = ("schema", "method", "config_sha256", "data_sha256",
             "seeds", "timing_ms", "results")
TIMING_STAGES = ("estimator", "candidate_screen", "policy", "mpc",
                 "plant_integration", "full_period")
PERCENTILES = ("p50", "p95", "p99", "max")
RESULT_FIELDS = ("episodes", "strict_hist_completed", "strict_mainline_completed",
                 "completed_with_backup", "fallback_episodes",
                 "empty_candidate_episodes", "first_violations")


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def validate(record):
    if tuple(record) != TOP_LEVEL or record["schema"] != SCHEMA:
        raise ValueError("run.json top-level schema or order differs")
    if not isinstance(record["method"], str) or not record["method"]:
        raise ValueError("method is required")
    for group in ("config_sha256", "data_sha256"):
        value = record[group]
        if not isinstance(value, dict) or not value or not all(
                isinstance(name, str) and _is_sha(digest)
                for name, digest in value.items()):
            raise ValueError(f"invalid {group}")
    if (not isinstance(record["seeds"], list) or not record["seeds"] or
            any(not isinstance(seed, int) or isinstance(seed, bool)
                for seed in record["seeds"])):
        raise ValueError("seeds must be integer identifiers")
    timings = record["timing_ms"]
    if tuple(timings) != TIMING_STAGES:
        raise ValueError("timing stage schema differs")
    for stage, values in timings.items():
        if values is None:
            continue
        if (tuple(values) != PERCENTILES or
                any(not isinstance(number, (int, float)) or
                    not math.isfinite(number) or number < 0
                    for number in values.values())):
            raise ValueError(f"invalid timing stage {stage}")
        if not (values["p50"] <= values["p95"] <= values["p99"] <= values["max"]):
            raise ValueError(f"nonmonotone timing percentiles for {stage}")
    results = record["results"]
    if tuple(results) != RESULT_FIELDS:
        raise ValueError("result schema differs")
    episodes = results["episodes"]
    if not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("positive episode count required")
    for field in ("strict_hist_completed", "strict_mainline_completed",
                  "completed_with_backup", "fallback_episodes",
                  "empty_candidate_episodes"):
        value = results[field]
        if value is not None and (not isinstance(value, int) or
                                  isinstance(value, bool) or not 0 <= value <= episodes):
            raise ValueError(f"invalid result field {field}")
    if not isinstance(results["first_violations"], dict):
        raise ValueError("first_violations must be a count mapping")
    if any(not isinstance(count, int) or count < 0
           for count in results["first_violations"].values()):
        raise ValueError("invalid first violation count")
    if results["strict_mainline_completed"] is not None:
        if (any(timings[stage] is None for stage in TIMING_STAGES) or
                results["completed_with_backup"] is None or
                results["empty_candidate_episodes"] is None):
            raise ValueError("formal mainline result lacks controller events or timings")
    return record


def write_once(path, record):
    validate(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text() != raw:
        raise RuntimeError(f"refusing to replace different run record: {path}")
    path.write_text(raw)
