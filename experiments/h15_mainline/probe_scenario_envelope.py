"""Measure dev residuals outside the frozen five-scenario E0 prediction envelope."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.h15_mainline.calibrate_response_intervals import replay
from experiments.h15_mainline.candidate_filter import _scenarios
from simulator.model import LoopPath, rotation_ned


ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (1, 5, 25, 35)
FIELDS = ("lateral_m", "depth_m", "pitch_deg", "tangent_water_speed_mps")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_values(state, tangent):
    normal = np.array([-tangent[1], tangent[0], 0.])
    normal /= max(np.linalg.norm(normal), 1e-12)
    water_speed = tangent @ (rotation_ned(state[3], state[4]) @ state[5:8])
    return np.array([normal @ state[:3], state[2],
                     math.degrees(state[3]), water_speed], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    collection = json.loads((args.collection / "run.json").read_text())
    if collection["scope"] != "dev only; no test observation files":
        raise ValueError("expected frozen dev-only collection")
    checkpoint = json.loads(args.checkpoint.read_text())
    manifest_path = ROOT / "experiments/h15_mainline/manifests/dev.json"
    if sha(manifest_path) != collection["dev_manifest_sha256"]:
        raise ValueError("dev manifest changed")
    manifest = json.loads(manifest_path.read_text())
    parameters = checkpoint["parameters"]
    gain, tau = parameters["e0_gain"], parameters["e0_tau_scale"]
    estimate = SimpleNamespace(
        fin_gain=gain, fin_tau_scale=tau,
        gain_radius=max(gain-parameters["gain_bounds"][0],
                        parameters["gain_bounds"][1]-gain),
        tau_radius=max(tau-parameters["tau_scale_bounds"][0],
                       parameters["tau_scale_bounds"][1]-tau))
    scenarios = _scenarios(estimate, parameters)
    summaries = json.loads((args.collection / "summary.json").read_text())
    expected = {(case["id"], seed) for case in manifest["cases"]
                for seed in manifest["seeds"]}
    if {(item["case_id"], item["seed"]) for item in summaries} != expected:
        raise ValueError("dev collection incomplete or duplicated")
    samples = defaultdict(lambda: defaultdict(list))
    trace_hashes = {}
    for episode in summaries:
        trace = ROOT / episode["trace"]
        if sha(trace) != episode["trace_sha256"]:
            raise ValueError(f"changed trace: {trace}")
        trace_hashes[episode["trace"]] = sha(trace)
        with trace.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        case = next(item for item in manifest["cases"]
                    if item["id"] == episode["case_id"])
        path = LoopPath(ROOT / case["path"])
        for start in range(0, len(rows), 5):
            if start + max(HORIZONS) > len(rows):
                continue
            commands = [json.loads(row["command_json"])
                        for row in rows[start:start + max(HORIZONS)]]
            predicted = [replay(rows[start], commands, gain_value, tau_value)
                         for _, gain_value, tau_value in scenarios]
            for horizon in HORIZONS:
                actual_row = rows[start + horizon - 1]
                actual = np.asarray(json.loads(actual_row["state_after_json"]))
                tangent = path.tangent_at(float(actual_row["progress_m"]))
                actual_values = physical_values(actual, tangent)
                values = np.stack([physical_values(states[horizon - 1], tangent)
                                   for states in predicted])
                low, high = values.min(axis=0), values.max(axis=0)
                outside = np.maximum(np.maximum(low-actual_values,
                                                actual_values-high), 0.)
                for index, field in enumerate(FIELDS):
                    samples[horizon][field + "_outside"].append(float(outside[index]))
                    samples[horizon][field + "_point"].append(
                        float(abs(values[0, index]-actual_values[index])))
                    samples[horizon][field + "_width"].append(float(high[index]-low[index]))
    result = {"schema": 1, "scope": "dev E0 scenario-envelope diagnostic, not held-out coverage",
              "method": "five_scenario_envelope_outside_residual",
              "scenarios": scenarios, "checkpoint_sha256": sha(args.checkpoint),
              "dev_manifest_sha256": sha(manifest_path),
              "dev_trace_sha256": trace_hashes, "quantile": .99,
              "statistics": {str(h): {field: {
                  "samples": len(samples[h][field + "_outside"]),
                  "point_abs_q99": float(np.quantile(samples[h][field + "_point"], .99)),
                  "outside_q99": float(np.quantile(samples[h][field + "_outside"], .99)),
                  "outside_fraction": float(np.mean(np.asarray(
                      samples[h][field + "_outside"]) > 0)),
                  "envelope_width_median": float(np.median(
                      samples[h][field + "_width"])),
                  "envelope_width_p99": float(np.quantile(
                      samples[h][field + "_width"], .99))}
                  for field in FIELDS} for h in HORIZONS}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({h: {field: data["outside_q99"]
                           for field, data in fields.items()}
                      for h, fields in result["statistics"].items()}, indent=2))


if __name__ == "__main__":
    main()
