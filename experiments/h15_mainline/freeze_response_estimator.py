"""Freeze the E_selected deterministic estimator and train/dev inputs."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from experiments.h15_mainline.response_estimator import SlidingWindowResponseEstimator


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != raw:
        raise RuntimeError(f"refusing changed frozen artifact: {path}")
    path.write_text(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    normalization = json.loads(args.normalization.read_text())
    calibration = json.loads(args.calibration.read_text())
    if normalization.get("split") != "train":
        raise RuntimeError("normalization must use train split")
    if calibration.get("method") != "empirical_dev_replay_residual_quantile":
        raise RuntimeError("unexpected calibration method")
    dev_manifest = ROOT / "experiments/h15_mainline/manifests/dev.json"
    if calibration["dev_manifest_sha256"] != sha(dev_manifest):
        raise RuntimeError("dev manifest changed since calibration")
    # No learned network weights: the checkpoint freezes the bounded WLS fit,
    # feature statistics and empirical constraint residual intervals.
    checkpoint = {"schema": 1, "method": "E_selected_E1_bounded_window_wls",
                  "parameters": SlidingWindowResponseEstimator.DEFAULTS,
                  "normalization_sha256": sha(args.normalization),
                  "calibration_sha256": sha(args.calibration),
                  "normalization_path": str(args.normalization.resolve().relative_to(ROOT)),
                  "calibration_path": str(args.calibration.resolve().relative_to(ROOT)),
                  "train_manifest_sha256": normalization["train_manifest_sha256"],
                  "dev_manifest_sha256": calibration["dev_manifest_sha256"],
                  "source_sha256": sha(ROOT / "experiments/h15_mainline/response_estimator.py"),
                  "claims": ["short-window equivalent response only",
                             "empirical dev residual intervals; no formal coverage or safety guarantee"]}
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out / "checkpoint.json"
    write_once(checkpoint_path, checkpoint)
    record = {"schema": 1, "method": "freeze_E_selected_E1",
              "checkpoint_sha256": sha(checkpoint_path),
              "normalization_sha256": sha(args.normalization),
              "calibration_sha256": sha(args.calibration),
              "source_sha256": {p: sha(ROOT / p) for p in (
                  "experiments/h15_mainline/response_estimator.py",
                  "experiments/h15_mainline/freeze_response_estimator.py")},
              "python_executable": "/bin/python3", "python": platform.python_version(),
              "seeds": {"train": [31001, 31002, 31003],
                        "dev": [32001, 32002, 32003]},
              "result": "frozen deterministic estimator checkpoint"}
    write_once(args.out / "run.json", record)
    (args.out / "console.log").write_text("froze E_selected E1 checkpoint from train/dev only\n")
    print(f"checkpoint sha256: {sha(checkpoint_path)}")


if __name__ == "__main__":
    main()
