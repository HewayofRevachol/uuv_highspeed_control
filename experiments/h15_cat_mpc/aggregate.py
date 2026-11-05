"""Aggregate per-episode CAT-MPC logs without pooling periods as samples."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    episodes = [json.loads(path.read_text()) for path in
                sorted(args.runs.glob("*/episode.json"))]
    summary = {"schema": 1, "episodes": len(episodes), "methods": {}}
    for method in sorted({row["method"] for row in episodes}):
        selected = [row for row in episodes if row["method"] == method]
        period_times = [time for row in selected for time in row["period_ms"]]
        steps = [step for row in selected for step in row["strict_substeps"]]
        violations = {kind: sum(bool(s["physical_flags"][kind]) for s in steps)
                      for kind in ("lateral", "depth", "speed", "pitch")}
        percentile = lambda values: ({"p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": float(max(values))}
            if values else None)
        timing_sources = {
            "adapter": [x.get("adapter_ms", 0.) for r in selected for x in r["alpha_log"]],
            "predictor": [x.get("predictor_ms", 0.) for r in selected for x in r["alpha_log"]],
            "policy": [x.get("policy_ms", 0.) for r in selected for x in r["alpha_log"]],
            "mpc": [x.get("timing", {}).get("mpc_wall_ms", 0.)
                    for r in selected for x in r["trace"]],
            "plant_integration": [x.get("timing", {}).get("plant_integration_wall_ms", 0.)
                                  for r in selected for x in r["trace"]],
        }
        numeric = [(abs(float(s["lateral_error_m"])), abs(float(s["depth_error_m"])),
                    abs(float(s["speed_error_mps"])),
                    abs(float(np.rad2deg(s["pitch_rad"])))) for s in steps]
        alpha = np.asarray([x["alpha"] for r in selected for x in r["alpha_log"]])
        raw_z = [x["raw_z"] for r in selected for x in r["alpha_log"]
                 if x.get("raw_z") is not None]
        summary["methods"][method] = {
            "episode_count": len(selected),
            "completed": sum(row["success"] for row in selected),
            "strict_hist_completed": sum(row["strict_hist"] for row in selected),
            "physical_strict_completed": sum(row["physical_strict"] for row in selected),
            "no_fallback_physical_completed": sum(row["no_fallback_physical"] for row in selected),
            "deadline_clean_completed": sum(row["deadline_clean"] for row in selected),
            "strict_mainline_completed": sum(row["strict_mainline"] for row in selected),
            "failed": sum(row["failed"] for row in selected),
            "fallback_periods": sum(row["fallback_periods"] for row in selected),
            "backup_periods": (sum(row["backup_periods"] for row in selected)
                               if all(row.get("backup_periods") is not None
                                      for row in selected) else None),
            "deadline_miss_periods": sum(sum(row["deadline_miss"]) for row in selected),
            "physical_violations_0p02s": violations,
            "completion_time_s_median": float(np.median(
                [row["periods"]*.1 for row in selected if row["success"]]))
                if any(row["success"] for row in selected) else None,
            "peak_errors": ({"lateral_m": max(x[0] for x in numeric),
                             "depth_m": max(x[1] for x in numeric),
                             "speed_mps": max(x[2] for x in numeric),
                             "pitch_deg": max(x[3] for x in numeric)} if numeric else None),
            "mean_abs_tangential_speed_error_mps": float(np.mean(
                [abs(float(s["speed_error_mps"])) for s in steps])) if steps else None,
            "timing_ms": {key: percentile(value) for key, value in timing_sources.items()},
            "period_ms": ({"p50": float(np.percentile(period_times, 50)),
                           "p95": float(np.percentile(period_times, 95)),
                           "p99": float(np.percentile(period_times, 99)),
                           "max": float(max(period_times))} if period_times else None),
            "progress_fraction_median": float(np.median(
                [row["progress_fraction"] for row in selected])),
            "alpha_mean": np.mean([entry["alpha"] for row in selected
                                   for entry in row["alpha_log"]], axis=0).tolist(),
            "alpha_std": alpha.std(0).tolist() if len(alpha) else None,
            "raw_z_std": np.std(raw_z, axis=0).tolist() if raw_z else None,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
