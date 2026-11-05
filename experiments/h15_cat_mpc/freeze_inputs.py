"""Freeze the completed P1 adapter/predictor artifacts for PPO rollouts."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--p1-analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.p1_analysis.read_text())
    if analysis.get("episodes") != 30 or analysis.get("paired_units") != 6:
        raise RuntimeError("frozen pilot inputs require all 30 P1 dev episodes")
    record = {"schema": 1, "status": "frozen P1 inputs",
        "normalization": str(args.normalization), "normalization_sha256": sha(args.normalization),
        "adapter": str(args.adapter), "adapter_sha256": sha(args.adapter),
        "predictor": str(args.predictor), "predictor_sha256": sha(args.predictor),
        "p1_analysis": str(args.p1_analysis), "p1_analysis_sha256": sha(args.p1_analysis),
        "P1_dev_residual_prediction_mae": analysis["residual_prediction_mae"],
        "train_manifest_sha256": sha(Path("experiments/h15_mainline/manifests/train.json")),
        "dev_manifest_sha256": sha(Path("experiments/h15_mainline/manifests/dev.json")),
        "holdout_used": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
