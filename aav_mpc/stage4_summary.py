"""Aggregate paired closed-loop metrics without counting failed runs in RMSE.

The bootstrap resamples paired path/seed units, never individual control
periods. These small samples are mechanism screens, not publication estimates.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.model import LoopPath


def quantile(values, q):
    return float(np.quantile(values, q)) if len(values) else None


def one(file):
    meta = json.loads(file.read_text())
    trace = file.with_suffix(".csv")
    if not trace.exists():
        return None
    with trace.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return None
    path = LoopPath(ROOT / meta["path"])
    offset = np.array([np.array(json.loads(row["state_after_json"]))[:3] -
        path.position_at(float(row["progress_m"])) for row in rows])
    error = np.linalg.norm(offset, axis=1)
    speed = np.abs(np.array([float(row["speed_error_mps"]) for row in rows]))
    wall = np.array([float(row["control_period_wall_ms"]) for row in rows])
    thrust = np.maximum(0, np.array([float(row["thrust_actual_n"]) for row in rows]))
    body_speed = np.array([float(row["body_speed_mps"]) for row in rows])
    return {"task": meta["task"], "method": meta["method"], "path": meta["path"],
        "seed": meta["seed"], "simulator_success": meta["simulator_success"],
        "strict_success": meta["strict_success"], "failed": meta["simulator_failed"],
        "first_anomaly": meta["first_anomaly"], "steps": len(rows),
        "position_rmse_m": float(np.sqrt(np.mean(error**2))),
        "position_p95_m": quantile(error, .95),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed**2))),
        "speed_p95_mps": quantile(speed, .95),
        "energy_proxy_j": float(np.sum(thrust * body_speed) * .1),
        "saturation_fraction": meta["saturation_fraction"],
        "fallback_fraction": meta["fallback_count"] / len(rows),
        "wall_p95_ms": quantile(wall, .95), "wall_p99_ms": quantile(wall, .99)}


def paired_ci(left, right, name, rng, repeats=4000):
    keys = sorted(set(left) & set(right))
    if not keys:
        return None
    difference = np.array([float(right[key][name])-float(left[key][name]) for key in keys])
    draws = rng.choice(difference, (repeats, len(difference)), replace=True).mean(1)
    return {"pairs": len(keys), "difference_mean": float(difference.mean()),
            "ci95": [quantile(draws, .025), quantile(draws, .975)]}


def summarize(directory):
    episodes = [one(file) for file in sorted(directory.glob("*.json"))]
    episodes = [row for row in episodes if row]
    by_task = defaultdict(list)
    for episode in episodes:
        by_task[episode["task"]].append(episode)
    report = {"episode_count": len(episodes), "tasks": {}}
    rng = np.random.default_rng(20261102)
    for task, rows in by_task.items():
        methods = defaultdict(list)
        for row in rows:
            methods[row["method"]].append(row)
        task_report = {"methods": {}, "paired_vs_osqp25": {},
                       "paired_vs_strong": {}}
        for method, sample in methods.items():
            safe = [r for r in sample if r["simulator_success"]]
            strict = [r for r in sample if r["strict_success"]]
            task_report["methods"][method] = {
                "episodes": len(sample), "simulator_success": len(safe),
                "strict_success": len(strict),
                "failure_types": dict(Counter("failed" if r["failed"] else
                    "timeout" if not r["simulator_success"] else
                    "task_violation" if not r["strict_success"] else "none" for r in sample)),
                "first_anomaly_types": dict(Counter(
                    r["first_anomaly"][1] if r["first_anomaly"] else "none"
                    for r in sample)),
                "all_episode_wall_p95_ms_mean": float(np.mean([r["wall_p95_ms"] for r in sample])),
                "all_episode_wall_p99_ms_mean": float(np.mean([r["wall_p99_ms"] for r in sample])),
                "all_episode_fallback_fraction_mean": float(np.mean([r["fallback_fraction"] for r in sample])),
                "common_success_only_metrics": {
                    key: float(np.mean([r[key] for r in safe])) if safe else None
                    for key in ("position_rmse_m", "position_p95_m", "speed_rmse_mps",
                                "speed_p95_mps", "steps", "energy_proxy_j",
                                "saturation_fraction", "wall_p95_ms", "wall_p99_ms")}}
        baseline = { (r["path"], r["seed"]): r for r in methods.get("osqp25", []) }
        for method, sample in methods.items():
            if method == "osqp25":
                continue
            counterpart = {(r["path"], r["seed"]): r for r in sample}
            paired = {
                metric: paired_ci(baseline, counterpart, metric, rng)
                for metric in ("simulator_success", "strict_success")}
            common = set(baseline) & set(counterpart)
            common_success = {key for key in common if baseline[key]["simulator_success"] and
                              counterpart[key]["simulator_success"]}
            paired["common_success_pairs"] = len(common_success)
            if common_success:
                paired["common_success_position_rmse_difference_m"] = float(np.mean([
                    counterpart[key]["position_rmse_m"] - baseline[key]["position_rmse_m"]
                    for key in common_success]))
                paired["common_success_speed_rmse_difference_mps"] = float(np.mean([
                    counterpart[key]["speed_rmse_mps"] - baseline[key]["speed_rmse_mps"]
                    for key in common_success]))
            task_report["paired_vs_osqp25"][method] = paired
        strong_name = "endpoint40" if task == "T1" else "curv6"
        strong = {(r["path"], r["seed"]): r for r in methods.get(strong_name, [])}
        for method, sample in methods.items():
            if method == strong_name:
                continue
            counterpart = {(r["path"], r["seed"]): r for r in sample}
            paired = {metric: paired_ci(strong, counterpart, metric, rng)
                      for metric in ("simulator_success", "strict_success")}
            common = {key for key in set(strong) & set(counterpart)
                      if strong[key]["simulator_success"] and counterpart[key]["simulator_success"]}
            paired["common_success_pairs"] = len(common)
            if common:
                left = {key: strong[key] for key in common}
                right = {key: counterpart[key] for key in common}
                for metric in ("position_rmse_m", "speed_rmse_mps", "energy_proxy_j",
                               "saturation_fraction", "steps", "wall_p95_ms"):
                    paired[metric] = paired_ci(left, right, metric, rng)
            task_report["paired_vs_strong"][method] = paired
        report["tasks"][task] = task_report
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("aav_mpc/data/stage4/paired"))
    parser.add_argument("--out", type=Path, default=Path("aav_mpc/data/stage4/summary.json"))
    args = parser.parse_args()
    report = summarize(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
