"""Freeze observation normalization using only the train collection."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.h15_mainline.observation_features import FEATURE_NAMES, from_period_row


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compensation-mode", choices=("on", "off"), default="on")
    args = parser.parse_args()
    run_path = args.collection / "run.json"
    run = json.loads(run_path.read_text())
    if run["scope"] != "train only; no dev or test observation files":
        raise RuntimeError("normalization requires train-only collection")
    modes = run.get("compensation_modes")
    if modes is None:
        modes = [run["compensation_mode"]] if run.get("compensation_mode") else [args.compensation_mode]
    if args.compensation_mode not in modes:
        raise RuntimeError("collection and requested compensation mode differ")
    manifest_path = ROOT / run["train_manifest"]
    if sha(manifest_path) != run["train_manifest_sha256"]:
        raise RuntimeError("training manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("split") != "T_fit":
        raise RuntimeError("normalization must be fitted on the supplemental T_fit split")
    summaries_path = args.collection / "summary.json"
    all_summaries = json.loads(summaries_path.read_text())
    summaries = [row for row in all_summaries
                 if row.get("compensation_mode", "on") == args.compensation_mode]
    expected = {(case["id"], seed) for case in manifest["cases"] for seed in manifest["seeds"]}
    observed = {(row["case_id"], row["seed"]) for row in summaries}
    if len(summaries) != len(expected) or observed != expected:
        raise RuntimeError("training collection is incomplete or duplicated")
    n = 0
    means = [0.] * len(FEATURE_NAMES)
    m2 = [0.] * len(FEATURE_NAMES)
    traces = {}
    for item in summaries:
        path = ROOT / item["trace"]
        if sha(path) != item["trace_sha256"]:
            raise RuntimeError(f"trace changed: {path}")
        traces[item["trace"]] = item["trace_sha256"]
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                values = from_period_row(row)
                n += 1
                for j, value in enumerate(values):
                    delta = value - means[j]
                    means[j] += delta / n
                    m2[j] += delta * (value - means[j])
    if n < 100:
        raise RuntimeError("too few training observations")
    scales = [max(math.sqrt(value / (n - 1)), 1e-6) for value in m2]
    result = {"schema": 2, "split": "train", "count": n,
              "compensation_mode": args.compensation_mode,
              "feature_names": FEATURE_NAMES, "mean": means, "scale": scales,
              "collection_run_sha256": sha(run_path),
              "collection_summary_sha256": sha(summaries_path),
              "train_manifest_sha256": sha(manifest_path), "trace_sha256": traces,
              "fit_method": "per-feature sample mean and sample standard deviation; 1e-6 floor",
              "excluded_inputs": ["target-domain true parameters", "future disturbance", "oracle labels"]}
    raw = json.dumps(result, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and args.out.read_text() != raw:
        raise RuntimeError("refusing to overwrite different frozen normalization")
    args.out.write_text(raw)
    run_record = {"schema": 2, "method": "h15_train_observation_normalization",
                  "compensation_mode": args.compensation_mode,
                  "input_collection_run_sha256": sha(run_path),
                  "input_collection_summary_sha256": sha(summaries_path),
                  "train_manifest_sha256": sha(manifest_path),
                  "normalization_sha256": sha(args.out),
                  "source_sha256": {
                      "experiments/h15_mainline/observation_features.py": sha(
                          ROOT / "experiments/h15_mainline/observation_features.py"),
                      "experiments/h15_mainline/fit_normalization.py": sha(Path(__file__))},
                  "python_executable": "/bin/python3"}
    (args.out.parent / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    (args.out.parent / "console.log").write_text(
        f"froze {len(FEATURE_NAMES)} features from {n} train periods\n")
    print(f"froze {len(FEATURE_NAMES)} features from {n} train periods")


if __name__ == "__main__":
    main()
