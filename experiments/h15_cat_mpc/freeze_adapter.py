"""Freeze train normalization and the deterministic existing response fit."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    norm = json.loads(args.normalization.read_text())
    if norm.get("split") != "train" or norm.get("count", 0) < 100:
        raise RuntimeError("adapter requires frozen train-only normalization")
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"schema": 1, "method": "existing_bounded_sliding_window_wls",
              "normalization": str(args.normalization),
              "normalization_sha256": sha(args.normalization),
              "estimator_source": "experiments/h15_mainline/response_estimator.py",
              "estimator_source_sha256": sha(ROOT/"experiments/h15_mainline/response_estimator.py"),
              "fallback_theta0": [.85, 2.], "window_periods": 10,
              "parameters": "SlidingWindowResponseEstimator.DEFAULTS",
              "split": "train", "formal_interval_calibration": False}
    (args.out/"frozen_adapter.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
