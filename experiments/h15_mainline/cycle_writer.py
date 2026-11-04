"""Write validated H15 period events and aggregate complete timing stages."""

import json
from pathlib import Path

import numpy as np

from experiments.h15_mainline.cycle_log import TIMING_FIELDS, validate
from experiments.h15_mainline.run_record import PERCENTILES


def write_once(path, records):
    """Write a complete ordered JSONL log; never overwrite different bytes."""
    path = Path(path)
    last = {}
    rows = []
    for record in records:
        validate(record)
        key = (record["method"], record["episode_id"])
        expected = last.get(key, -1) + 1
        if record["period"] != expected:
            raise ValueError(f"nonconsecutive period for {key}: {record['period']}")
        last[key] = record["period"]
        rows.append(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    if not rows:
        raise ValueError("cycle log cannot be empty")
    raw = "".join(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != raw:
        raise RuntimeError(f"refusing to replace different cycle log: {path}")
    path.write_text(raw)


def timing_summary(records):
    """Return p50/p95/p99/max by stage; unavailable stages remain null."""
    rows = list(records)
    if not rows:
        raise ValueError("cannot summarize empty cycle log")
    for row in rows:
        validate(row)
    result = dict.fromkeys(TIMING_FIELDS)
    for stage in TIMING_FIELDS:
        values = [row["timing_ms"][stage] for row in rows]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(f"partly missing timing stage {stage}")
        array = np.asarray(values, dtype=float)
        result[stage] = dict(zip(PERCENTILES,
                                 [float(np.percentile(array, p)) for p in (50, 95, 99)] +
                                 [float(array.max())]))
    return result
