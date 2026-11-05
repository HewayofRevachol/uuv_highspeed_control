"""Apply P2 exploration gates to PPO versus A0 and best deterministic gate."""
import argparse
import json
from pathlib import Path

import numpy as np


P2_PROTOCOL = {
    "version": "legacy_p2_audit_repaired_v2",
    "scope": "legacy P2 exploration only; not E4/E5 acceptance",
    "safety_precondition": "no paired physical/no-fallback regression; no paired fallback-count increase",
    "precision_set": "both success and no_fallback_physical",
    "relative_statistic": "median of per-pair relative changes",
    "zero_baseline": "0 to 0 is zero change; positive error from zero blocks precision rules",
    "thresholds": {"rule1_minimum_no_fallback_improvements": 2,
                   "rule2_rmse_relative_change_max": -.10,
                   "rule2_time_relative_change_max": .03,
                   "rule3_fallback_rate_ratio_max": .8},
}


def normalized_rmse(episode):
    values = []
    for row in episode["strict_substeps"]:
        values.append(np.sqrt((float(row["lateral_error_m"])/8.)**2+
            (float(row["depth_error_m"])/5.)**2+
            (float(row["speed_error_mps"])/2.)**2+
            (float(np.rad2deg(row["pitch_rad"]))/40.)**2))
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def speed_mae(episode):
    values = [abs(float(x["speed_error_mps"])) for x in episode["strict_substeps"]]
    return float(np.mean(values)) if values else None


def _relative_change(candidate, baseline):
    if candidate is None or baseline is None:
        return None
    if not np.isfinite([candidate, baseline]).all():
        return None
    if baseline == 0.:
        return 0. if candidate == 0. else None
    return float(candidate / baseline - 1.)


def _median_complete(rows, key):
    values = [row[key] for row in rows]
    if not values or any(value is None or not np.isfinite(value) for value in values):
        return None
    return float(np.median(values))


def summarize_pairs(candidate, baseline):
    if set(candidate) != set(baseline) or not baseline:
        raise ValueError("candidate and baseline must contain the same nonempty paired units")
    pairs = []
    for key in sorted(baseline):
        a, b = baseline[key], candidate[key]
        common = bool(a["success"] and b["success"] and
                      a["no_fallback_physical"] and b["no_fallback_physical"])
        baseline_rmse = normalized_rmse(a) if common else None
        candidate_rmse = normalized_rmse(b) if common else None
        pairs.append({"unit": key, "baseline_physical_strict": a["physical_strict"],
            "candidate_physical_strict": b["physical_strict"],
            "baseline_no_fallback_physical": a["no_fallback_physical"],
            "candidate_no_fallback_physical": b["no_fallback_physical"],
            "baseline_deadline_clean": a["deadline_clean"],
            "candidate_deadline_clean": b["deadline_clean"],
            "baseline_strict_mainline": a["strict_mainline"],
            "candidate_strict_mainline": b["strict_mainline"],
            "physical_improved": not a["physical_strict"] and b["physical_strict"],
            "physical_worsened": a["physical_strict"] and not b["physical_strict"],
            "no_fallback_improved": not a["no_fallback_physical"] and b["no_fallback_physical"],
            "no_fallback_worsened": a["no_fallback_physical"] and not b["no_fallback_physical"],
            "improved_to_unintervened_physical_finish":
                not a["no_fallback_physical"] and b["no_fallback_physical"],
            "worsened_from_unintervened_physical_finish":
                a["no_fallback_physical"] and not b["no_fallback_physical"],
            "deadline_improved": not a["deadline_clean"] and b["deadline_clean"],
            "deadline_worsened": a["deadline_clean"] and not b["deadline_clean"],
            "baseline_fallback_periods": a["fallback_periods"],
            "candidate_fallback_periods": b["fallback_periods"],
            "fallback_increased": b["fallback_periods"] > a["fallback_periods"],
            "common_no_fallback_success": common,
            "common_success": common,
            "baseline_normalized_rmse": baseline_rmse,
            "candidate_normalized_rmse": candidate_rmse,
            "normalized_rmse_relative_change": _relative_change(candidate_rmse, baseline_rmse),
            "normalized_rmse_absolute_change": (candidate_rmse-baseline_rmse
                if baseline_rmse is not None and candidate_rmse is not None else None),
            "completion_time_relative_change": _relative_change(b["periods"], a["periods"])
                if common else None,
            "baseline_time_s": a["periods"]*.1 if common else None,
            "candidate_time_s": b["periods"]*.1 if common else None,
            "baseline_speed_mae": speed_mae(a) if common else None,
            "candidate_speed_mae": speed_mae(b) if common else None})
    both = [x for x in pairs if x["common_success"]]
    base_interventions = sum(a["fallback_periods"] for a in baseline.values())
    cand_interventions = sum(candidate[k]["fallback_periods"] for k in baseline)
    base_periods = sum(a["periods"] for a in baseline.values())
    cand_periods = sum(candidate[k]["periods"] for k in baseline)
    base_rate = base_interventions/base_periods if base_periods else 0.
    cand_rate = cand_interventions/cand_periods if cand_periods else 0.
    physical_worsened = sum(x["physical_worsened"] for x in pairs)
    no_fallback_worsened = sum(x["no_fallback_worsened"] for x in pairs)
    fallback_increased = sum(x["fallback_increased"] for x in pairs)
    return {"protocol_version": P2_PROTOCOL["version"], "pairs": pairs,
        "improved": sum(x["improved_to_unintervened_physical_finish"] for x in pairs),
        "worsened": sum(x["worsened_from_unintervened_physical_finish"] for x in pairs),
        "baseline_strict_mainline": sum(x["baseline_strict_mainline"] for x in pairs),
        "candidate_strict_mainline": sum(x["candidate_strict_mainline"] for x in pairs),
        "baseline_physical_strict": sum(x["baseline_physical_strict"] for x in pairs),
        "candidate_physical_strict": sum(x["candidate_physical_strict"] for x in pairs),
        "baseline_no_fallback_physical": sum(x["baseline_no_fallback_physical"] for x in pairs),
        "candidate_no_fallback_physical": sum(x["candidate_no_fallback_physical"] for x in pairs),
        "physical_improved": sum(x["physical_improved"] for x in pairs),
        "physical_worsened": physical_worsened,
        "no_fallback_improved": sum(x["no_fallback_improved"] for x in pairs),
        "no_fallback_worsened": no_fallback_worsened,
        "deadline_improved": sum(x["deadline_improved"] for x in pairs),
        "deadline_worsened": sum(x["deadline_worsened"] for x in pairs),
        "common_successes": len(both),
        "common_no_fallback_successes": len(both),
        "baseline_normalized_rmse_median": _median_complete(both, "baseline_normalized_rmse"),
        "candidate_normalized_rmse_median": _median_complete(both, "candidate_normalized_rmse"),
        "normalized_rmse_relative_change": _median_complete(both, "normalized_rmse_relative_change"),
        "completion_time_relative_change": _median_complete(both, "completion_time_relative_change"),
        "baseline_speed_mae_median": _median_complete(both, "baseline_speed_mae"),
        "candidate_speed_mae_median": _median_complete(both, "candidate_speed_mae"),
        "paired_fallback_increases": fallback_increased,
        "safety_precondition": not (physical_worsened or no_fallback_worsened or fallback_increased),
        "backup_periods": None,
        "baseline_fallback_periods": base_interventions,
        "candidate_fallback_periods": cand_interventions,
        "baseline_fallback_rate": base_rate,
        "candidate_fallback_rate": cand_rate}


def exploration_rules(summary):
    """Legacy P2 thresholds with corrected physical and fallback prerequisites."""
    safe = summary["safety_precondition"]
    rmse = summary["normalized_rmse_relative_change"]
    duration = summary["completion_time_relative_change"]
    speeds = (summary["candidate_speed_mae_median"], summary["baseline_speed_mae_median"])
    speed_ok = all(value is not None for value in speeds) and speeds[0] <= speeds[1]
    return {"safety_precondition": safe,
        "rule1": bool(safe and summary["no_fallback_improved"] >= 2),
        "rule2": bool(safe and summary["common_no_fallback_successes"] > 0 and
            rmse is not None and rmse <= -.10 and duration is not None and
            duration <= .03 and speed_ok),
        "rule3": bool(safe and rmse is not None and rmse <= 0. and speed_ok and
            summary["baseline_fallback_rate"] > 0. and summary["candidate_fallback_rate"] <=
            .8*summary["baseline_fallback_rate"])}


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
        rule_results[checkpoint] = {**exploration_rules(a0),
            "versus_A4_no_fallback_improved": comparisons["A4"]["no_fallback_improved"],
            "versus_A4_no_fallback_worsened": comparisons["A4"]["no_fallback_worsened"]}
    result = {"schema": 2, "protocol": P2_PROTOCOL, "dev_units": len(baselines["A0"]),
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
