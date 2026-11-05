"""Apply P2 exploration gates to PPO versus A0 and best deterministic gate."""
import argparse
import json
from pathlib import Path

import numpy as np


def normalized_rmse(episode):
    values = []
    for row in episode["strict_substeps"]:
        values.append(np.sqrt((float(row["lateral_error_m"])/8.)**2+
            (float(row["depth_error_m"])/5.)**2+
            (float(row["speed_error_mps"])/2.)**2+
            (float(np.rad2deg(row["pitch_rad"]))/40.)**2))
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def speed_mae(episode):
    return float(np.mean([abs(float(x["speed_error_mps"]))
                          for x in episode["strict_substeps"]]))


def summarize_pairs(candidate, baseline):
    pairs = []
    for key in sorted(baseline):
        a, b = baseline[key], candidate[key]
        common = bool(a["success"] and b["success"])
        pairs.append({"unit": key, "baseline_physical_strict": a["physical_strict"],
            "candidate_physical_strict": b["physical_strict"],
            "baseline_deadline_clean": a["deadline_clean"],
            "candidate_deadline_clean": b["deadline_clean"],
            "baseline_strict_mainline": a["strict_mainline"],
            "candidate_strict_mainline": b["strict_mainline"],
            "improved_to_unintervened_physical_finish":
                not a["physical_strict"] and b["physical_strict"],
            "worsened_from_unintervened_physical_finish":
                a["physical_strict"] and not b["physical_strict"],
            "deadline_improved": not a["deadline_clean"] and b["deadline_clean"],
            "deadline_worsened": a["deadline_clean"] and not b["deadline_clean"],
            "common_success": common,
            "baseline_normalized_rmse": normalized_rmse(a) if common else None,
            "candidate_normalized_rmse": normalized_rmse(b) if common else None,
            "baseline_time_s": a["periods"]*.1 if common else None,
            "candidate_time_s": b["periods"]*.1 if common else None,
            "baseline_speed_mae": speed_mae(a) if common else None,
            "candidate_speed_mae": speed_mae(b) if common else None})
    both = [x for x in pairs if x["common_success"]]
    base_rmse = np.median([x["baseline_normalized_rmse"] for x in both]) if both else None
    cand_rmse = np.median([x["candidate_normalized_rmse"] for x in both]) if both else None
    base_time = np.median([x["baseline_time_s"] for x in both]) if both else None
    cand_time = np.median([x["candidate_time_s"] for x in both]) if both else None
    base_interventions = sum(a["fallback_periods"] for a in baseline.values())
    cand_interventions = sum(candidate[k]["fallback_periods"] for k in baseline)
    base_periods = sum(a["periods"] for a in baseline.values())
    cand_periods = sum(candidate[k]["periods"] for k in baseline)
    base_rate = base_interventions/base_periods if base_periods else 0.
    cand_rate = cand_interventions/cand_periods if cand_periods else 0.
    return {"pairs": pairs,
        "improved": sum(x["improved_to_unintervened_physical_finish"] for x in pairs),
        "worsened": sum(x["worsened_from_unintervened_physical_finish"] for x in pairs),
        "baseline_strict_mainline": sum(x["baseline_strict_mainline"] for x in pairs),
        "candidate_strict_mainline": sum(x["candidate_strict_mainline"] for x in pairs),
        "baseline_physical_strict": sum(x["baseline_physical_strict"] for x in pairs),
        "candidate_physical_strict": sum(x["candidate_physical_strict"] for x in pairs),
        "physical_improved": sum(x["improved_to_unintervened_physical_finish"] for x in pairs),
        "physical_worsened": sum(x["worsened_from_unintervened_physical_finish"] for x in pairs),
        "deadline_improved": sum(x["deadline_improved"] for x in pairs),
        "deadline_worsened": sum(x["deadline_worsened"] for x in pairs),
        "common_successes": len(both),
        "baseline_normalized_rmse_median": float(base_rmse) if both else None,
        "candidate_normalized_rmse_median": float(cand_rmse) if both else None,
        "normalized_rmse_relative_change": float(cand_rmse/base_rmse-1.) if both and base_rmse else None,
        "completion_time_relative_change": float(cand_time/base_time-1.) if both and base_time else None,
        "baseline_speed_mae_median": float(np.median([x["baseline_speed_mae"] for x in both])) if both else None,
        "candidate_speed_mae_median": float(np.median([x["candidate_speed_mae"] for x in both])) if both else None,
        "backup_periods": None,
        "baseline_fallback_rate": base_rate,
        "candidate_fallback_rate": cand_rate}


def load_runs(directory):
    runs = [json.loads(p.read_text()) for p in directory.glob("*/episode.json")]
    if any(row.get("command_schema") != "issued_v2" or row.get("schema", 0) < 2
           for row in runs):
        raise RuntimeError("refusing to mix legacy and issued_v2 episode logs")
    return {(Path(x["path"]).stem, x["seed"], x["method"]): x for x in runs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-runs", type=Path, required=True)
    parser.add_argument("--fixed-runs", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ppo = load_runs(args.ppo_runs)
    fixed = load_runs(args.fixed_runs)
    baselines = {method: {(key[0], key[1]): value for key, value in fixed.items()
                          if key[2] == method} for method in ("A0", "A4")}
    results = {}
    for checkpoint in ("0000", "1000", "2000", "4000"):
        candidate = {(key[0], key[1]): value for key, value in ppo.items()
                     if key[2] == "PPO_"+checkpoint}
        results[checkpoint] = {method: summarize_pairs(candidate, base)
                               for method, base in baselines.items()}
    training = json.loads(args.training_metrics.read_text())
    rule_results = {}
    for checkpoint, comparisons in results.items():
        a0 = comparisons["A0"]
        rule_results[checkpoint] = {
            "rule1": a0["improved"] >= 2 and a0["worsened"] == 0,
            "rule2": (a0["common_successes"] > 0 and
                a0["normalized_rmse_relative_change"] is not None and
                a0["normalized_rmse_relative_change"] <= -.10 and
                a0["completion_time_relative_change"] is not None and
                a0["completion_time_relative_change"] <= .03 and
                a0["candidate_speed_mae_median"] <= a0["baseline_speed_mae_median"]),
            "rule3": (a0["normalized_rmse_relative_change"] is not None and
                a0["normalized_rmse_relative_change"] <= 0. and
                a0["candidate_speed_mae_median"] <= a0["baseline_speed_mae_median"] and
                a0["baseline_fallback_rate"] > 0. and
                a0["candidate_fallback_rate"] <=
                    .8*a0["baseline_fallback_rate"]),
            "versus_A4_mainline_improved": comparisons["A4"]["improved"],
            "versus_A4_mainline_worsened": comparisons["A4"]["worsened"]}
    result = {"schema": 1, "dev_units": len(fixed)//5,
        "P2_exploration_rules": rule_results,
        "checkpoint_comparisons": results,
        "training_curve": training,
        "interpretation": "pilot only; unit of pairing is geometry x seed"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"rules": rule_results,
        "summary": {checkpoint: {method: {
            key: value for key, value in summary.items()
            if key != "pairs"} for method, summary in methods.items()}
            for checkpoint, methods in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
