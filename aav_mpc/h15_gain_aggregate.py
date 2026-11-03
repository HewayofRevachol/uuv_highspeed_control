"""Aggregate the frozen 0.9 gain + 1.69 fin-response diagnostic cells."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--holdout-ac", type=Path, required=True)
    parser.add_argument("--holdout-bd", type=Path, required=True)
    parser.add_argument("--baseline-target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_file = Path("data/datasets/h15_mpc_replication/manifest.json")
    manifest = json.loads(manifest_file.read_text())
    paths = {p["family"]: p for p in manifest["splits"]["replication"]
             if p["generator_family"] == "coupled"}
    expected = {(family, seed) for family in paths
                for seed in manifest["experiment_freeze"]["seeds"]}
    corrected = {}
    file_hashes = {}
    runs = (args.screen, args.holdout_ac, args.holdout_bd)
    for directory in runs:
        config_file = directory / "run.json"
        config = json.loads(config_file.read_text())
        if (config.get("prediction_fin_gain_scale") != .9 or
                config.get("prediction_fin_tau_scale") != 1.69 or
                config.get("variants") != ["full"]):
            parser.error(f"run is not the specified full-domain diagnostic: {directory}")
        file_hashes[str(config_file)] = digest(config_file)
        for file in directory.glob("replication_*_condensed_endpoint35_full_gain09_tau169.json"):
            case = json.loads(file.read_text())
            family = Path(case["path"]).stem.removeprefix("replication_")
            key = (family, case["seed"])
            if (key not in expected or key in corrected or
                    case["path"] != paths[family]["path"] or
                    case["method"] != "condensed_endpoint35" or
                    case["target_components"] != ["fin", "integrator", "actuator"] or
                    case["prediction_fin_gain_scale"] != .9 or
                    case["prediction_fin_tau_scale"] != 1.69):
                raise ValueError(f"duplicate or mismatched diagnostic case: {file}")
            corrected[key] = case
            file_hashes[str(file)] = digest(file)
    if set(corrected) != expected:
        raise ValueError(f"missing paired cases: {sorted(expected - set(corrected))}")
    baseline = {}
    for family, seed in expected:
        file = args.baseline_target / f"replication_{family}_s{seed}_condensed_endpoint35_target.json"
        case = json.loads(file.read_text())
        if (case["path"] != paths[family]["path"] or case["seed"] != seed or
                case["method"] != "condensed_endpoint35" or not case["target_domain"]):
            raise ValueError(f"mismatched baseline case: {file}")
        baseline[(family, seed)] = case
        file_hashes[str(file)] = digest(file)
    pairs = [(baseline[key], corrected[key]) for key in sorted(expected)]
    common = [(a, b) for a, b in pairs if a["strict_success"] and b["strict_success"]]
    summary = {
        "pairs": len(pairs),
        "baseline_success": sum(a["strict_success"] for a, b in pairs),
        "corrected_success": sum(b["strict_success"] for a, b in pairs),
        "improved": sum(not a["strict_success"] and b["strict_success"] for a, b in pairs),
        "worsened": sum(a["strict_success"] and not b["strict_success"] for a, b in pairs),
        "common_success_pairs": len(common),
        "common_success_rmse_delta_m": (mean(b["position_rmse_m"]-a["position_rmse_m"]
                                              for a, b in common) if common else None),
        "corrected_fallback_count": sum(b["fallback_count"] for a, b in pairs),
        "corrected_first_anomaly_types": dict(Counter(
            (b["first_anomaly"] or [0, "none"])[1] for a, b in pairs)),
        "by_family": {family: {
            "baseline_success": sum(baseline[(family, seed)]["strict_success"]
                                    for seed in manifest["experiment_freeze"]["seeds"]),
            "corrected_success": sum(corrected[(family, seed)]["strict_success"]
                                     for seed in manifest["experiment_freeze"]["seeds"]),
        } for family in paths},
    }
    output = {"manifest_sha256": digest(manifest_file),
              "timing_note": "Holdout subruns were concurrent; retime serially before any realtime claim.",
              "source_sha256": {str(Path(__file__).relative_to(Path.cwd())): digest(Path(__file__))},
              "input_sha256": file_hashes, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
