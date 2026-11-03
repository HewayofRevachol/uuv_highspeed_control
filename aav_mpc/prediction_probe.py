"""Replay issued closed-loop commands from measured states for 1/5/25-step errors.

The 10-state comparator applies each issued command without lag. The
actuator-aware predictor uses declared mismatch parameters and observed
actuator/history. Both see exactly the same future issued commands. This
isolates dynamics prediction, not policy counterfactual behavior.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.model import DT, MISMATCHED_PLANT, NOMINAL_PLANT, rk4_step


def propagate(initial, actual, queue, commands, aware):
    x = np.array(initial, dtype=float)
    a = np.array(actual, dtype=float)
    history = np.array(queue, dtype=float)
    p = MISMATCHED_PLANT
    alpha = np.array([-np.expm1(-DT / p.thrust_tau_s),
                      -np.expm1(-DT / p.fin_tau_s),
                      -np.expm1(-DT / p.fin_tau_s)])
    estimates = []
    for command in commands:
        for _ in range(5):
            if aware:
                target = history[2].copy()
                history = np.vstack((command, history[:-1]))
                a += alpha * (target - a)
                x = rk4_step(x, a, plant_params=p)
            else:
                x = rk4_step(x, command, plant_params=NOMINAL_PLANT)
        estimates.append(x.copy())
    return np.array(estimates)


def evaluate(file, stride=25):
    with file.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = ("state_before_json", "actuator_before_json", "queue_before_json",
                "command_json", "state_after_json")
    if not rows or not all(name in rows[0] for name in required):
        raise ValueError("trace must contain full measured states and issued commands")
    commands = np.array([json.loads(row["command_json"]) for row in rows])
    actual_states = np.array([json.loads(row["state_after_json"]) for row in rows])
    metrics = {model: {h: [] for h in (1, 5, 25)} for model in ("nominal_10_state", "actuator_aware_22_state")}
    for i in range(0, len(rows)-25, stride):
        before = json.loads(rows[i]["state_before_json"])
        actual = json.loads(rows[i]["actuator_before_json"])
        queue = json.loads(rows[i]["queue_before_json"])
        for name, aware in (("nominal_10_state", False), ("actuator_aware_22_state", True)):
            predicted = propagate(before, actual, queue, commands[i:i+25], aware)
            for h in (1, 5, 25):
                difference = predicted[h-1] - actual_states[i+h-1]
                metrics[name][h].append([np.linalg.norm(difference[:3]),
                                         np.linalg.norm(difference[5:8]),
                                         abs(difference[3]), abs(difference[4])])
    result = {"trace": str(file), "start_count": len(metrics["nominal_10_state"][1]),
              "stride": stride, "errors": {}}
    for name, horizons in metrics.items():
        result["errors"][name] = {}
        for h, values in horizons.items():
            matrix = np.array(values)
            result["errors"][name][str(h)] = {
                key: {"median": float(np.median(matrix[:, j])),
                      "p95": float(np.quantile(matrix[:, j], .95))}
                for j, key in enumerate(("position_m", "velocity_mps", "pitch_rad", "yaw_rad"))}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.trace, args.stride)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
