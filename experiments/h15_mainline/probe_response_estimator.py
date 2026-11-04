"""Measure E1 activation, bounded output and runtime on train observations."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.h15_mainline.response_estimator import SlidingWindowResponseEstimator


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.collection / "run.json").read_text())
    if run["scope"] != "train only; no dev or test observation files":
        raise RuntimeError("probe only accepts train collection")
    normalization = json.loads(args.normalization.read_text())
    if normalization["collection_run_sha256"] != sha(args.collection / "run.json"):
        raise RuntimeError("normalization does not match collection")
    summaries = json.loads((args.collection / "summary.json").read_text())
    output_rows = []
    trace_hashes = {}
    for episode in summaries:
        path = ROOT / episode["trace"]
        if sha(path) != episode["trace_sha256"]:
            raise RuntimeError(f"trace changed: {path}")
        trace_hashes[episode["trace"]] = sha(path)
        estimator = SlidingWindowResponseEstimator(normalization)
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                result = estimator.update(row)
                output_rows.append({"case_id": episode["case_id"],
                                    "seed": episode["seed"], "step": row["step"],
                                    "source": result.source,
                                    "fallback_reason": result.fallback_reason or "",
                                    "fin_gain": result.fin_gain,
                                    "fin_tau_scale": result.fin_tau_scale,
                                    "gain_radius": result.gain_radius,
                                    "tau_radius": result.tau_radius,
                                    "forward_ms": result.forward_ms,
                                    "fallback_ms": result.fallback_ms,
                                    "history_samples": result.samples})
    args.out.mkdir(parents=True, exist_ok=True)
    trace_out = args.out / "estimator_trace.csv"
    with trace_out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0])
        writer.writeheader()
        writer.writerows(output_rows)
    counts = Counter((row["source"], row["fallback_reason"]) for row in output_rows)
    timing = np.asarray([row["forward_ms"] + row["fallback_ms"] for row in output_rows])
    report = {"schema": 1, "scope": "train-only estimator probe; intervals uncalibrated",
              "periods": len(output_rows), "episodes": len(summaries),
              "counts": {f"{source}:{reason or 'active'}": count
                         for (source, reason), count in sorted(counts.items())},
              "timing_ms": {f"p{p}": float(np.percentile(timing, p))
                            for p in (50, 95, 99)} | {"max": float(timing.max())},
              "observed_gain_range": [min(row["fin_gain"] for row in output_rows),
                                      max(row["fin_gain"] for row in output_rows)],
              "observed_tau_scale_range": [min(row["fin_tau_scale"] for row in output_rows),
                                          max(row["fin_tau_scale"] for row in output_rows)]}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    run_record = {"schema": 1, "method": "h15_e1_train_probe",
                  "collection_run_sha256": sha(args.collection / "run.json"),
                  "normalization_sha256": sha(args.normalization),
                  "trace_sha256": trace_hashes,
                  "source_sha256": {p: sha(ROOT / p) for p in (
                      "experiments/h15_mainline/response_estimator.py",
                      "experiments/h15_mainline/probe_response_estimator.py")},
                  "python_executable": "/bin/python3",
                  "result_files": {"estimator_trace.csv": sha(trace_out),
                                   "summary.json": sha(args.out / "summary.json")}}
    (args.out / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    (args.out / "console.log").write_text(
        f"probed {len(summaries)} train episodes, {len(output_rows)} periods\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
