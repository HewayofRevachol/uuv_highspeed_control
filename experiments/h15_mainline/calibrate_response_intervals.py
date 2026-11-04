"""Calibrate E1/E0 empirical replay residual radii on frozen dev episodes."""

import argparse
from collections import Counter
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.h15_mainline.response_estimator import SlidingWindowResponseEstimator
from simulator.model import MISMATCHED_PLANT, LoopPath, rk4_step, rotation_ned


ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (1, 5, 25, 35)
FIELDS = ("lateral_m", "depth_m", "pitch_deg", "tangent_water_speed_mps")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(start_row, commands, gain, tau_scale):
    """Nominal physical replay with observed future commands; dev-only use."""
    state = np.asarray(json.loads(start_row["state_before_json"]), dtype=float)
    actuator = np.asarray(json.loads(start_row["actuator_before_json"]), dtype=float)
    queue = np.asarray(json.loads(start_row["queue_before_json"]), dtype=float)[:3]
    config = replace(MISMATCHED_PLANT,
                     elevator_gain=MISMATCHED_PLANT.elevator_gain*gain,
                     rudder_gain=MISMATCHED_PLANT.rudder_gain*gain)
    delays = (math.ceil(MISMATCHED_PLANT.thrust_delay_s/.02-1e-12),
              math.ceil(MISMATCHED_PLANT.fin_delay_s/.02-1e-12),
              math.ceil(MISMATCHED_PLANT.fin_delay_s/.02-1e-12))
    taus = (MISMATCHED_PLANT.thrust_tau_s,
            MISMATCHED_PLANT.fin_tau_s*tau_scale,
            MISMATCHED_PLANT.fin_tau_s*tau_scale)
    alpha = np.asarray([-math.expm1(-.02/tau) if tau else 1. for tau in taus])
    states = []
    for command in commands:
        command = np.asarray(command, dtype=float)
        for _ in range(5):
            target = np.asarray([queue[d-1, j] if d else command[j]
                                 for j, d in enumerate(delays)])
            queue = np.vstack((command, queue[:-1]))
            actuator += alpha*(target-actuator)
            state = rk4_step(state, actuator, plant_params=config)
        states.append(state.copy())
    return states


def errors(predicted, actual_row, path):
    actual = np.asarray(json.loads(actual_row["state_after_json"]), dtype=float)
    tangent = path.tangent_at(float(actual_row["progress_m"]))
    difference = predicted[:3]-actual[:3]
    horizontal_normal = np.asarray([-tangent[1], tangent[0], 0.])
    horizontal_normal /= max(np.linalg.norm(horizontal_normal), 1e-12)
    speed_pred = tangent @ (rotation_ned(predicted[3], predicted[4]) @ predicted[5:8])
    speed_actual = tangent @ (rotation_ned(actual[3], actual[4]) @ actual[5:8])
    return (abs(float(horizontal_normal@difference)), abs(float(difference[2])),
            abs(math.degrees(float(predicted[3]-actual[3]))),
            abs(float(speed_pred-speed_actual)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collection_run = json.loads((args.collection / "run.json").read_text())
    if collection_run["scope"] != "dev only; no test observation files":
        raise RuntimeError("calibration requires dev-only collection")
    normalization = json.loads(args.normalization.read_text())
    if normalization["split"] != "train":
        raise RuntimeError("normalization must be train-only")
    summaries = json.loads((args.collection / "summary.json").read_text())
    manifest_path = ROOT / "experiments/h15_mainline/manifests/dev.json"
    if sha(manifest_path) != collection_run["dev_manifest_sha256"]:
        raise RuntimeError("dev manifest differs")
    manifest = json.loads(manifest_path.read_text())
    expected = {(case["id"], seed) for case in manifest["cases"] for seed in manifest["seeds"]}
    if len(summaries) != len(expected) or {(s["case_id"], s["seed"]) for s in summaries} != expected:
        raise RuntimeError("dev collection incomplete or duplicated")
    residuals = {h: {field: [] for field in FIELDS} for h in HORIZONS}
    source_counts = Counter()
    input_hashes = {}
    for episode in summaries:
        trace = ROOT / episode["trace"]
        if sha(trace) != episode["trace_sha256"]:
            raise RuntimeError(f"changed dev trace: {trace}")
        input_hashes[episode["trace"]] = sha(trace)
        path_entry = next(case for case in manifest["cases"] if case["id"] == episode["case_id"])
        path = LoopPath(ROOT / path_entry["path"])
        with trace.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        estimator = SlidingWindowResponseEstimator(normalization)
        estimates = [estimator.update(row) for row in rows]
        source_counts.update(e.source for e in estimates)
        for i in range(0, len(rows), 5):
            if i+max(HORIZONS) > len(rows):
                continue
            estimate = estimates[i]
            commands = [json.loads(row["command_json"]) for row in rows[i:i+max(HORIZONS)]]
            predictions = replay(rows[i], commands, estimate.fin_gain,
                                 estimate.fin_tau_scale)
            for h in HORIZONS:
                values = errors(predictions[h-1], rows[i+h-1], path)
                for field, value in zip(FIELDS, values):
                    if not math.isfinite(value):
                        raise RuntimeError("nonfinite replay residual")
                    residuals[h][field].append(value)
    raw_radii = {str(h): {field: float(np.quantile(values, .99))
                           for field, values in residuals[h].items()} for h in HORIZONS}
    radii = {}
    previous = {field: 0. for field in FIELDS}
    for h in HORIZONS:
        radii[str(h)] = {field: max(previous[field], raw_radii[str(h)][field])
                         for field in FIELDS}
        previous = radii[str(h)]
    counts = {str(h): len(residuals[h][FIELDS[0]]) for h in HORIZONS}
    result = {"schema": 1, "method": "empirical_dev_replay_residual_quantile",
              "quantile": .99, "horizons_control_periods": HORIZONS,
              "fields": FIELDS, "radii": radii, "raw_radii": raw_radii,
              "radius_rule": "99th percentile, nondecreasing envelope across horizons",
              "sample_counts": counts,
              "estimator_source_counts": dict(source_counts),
              "scope": "in-sample dev calibration, no formal coverage or safety guarantee",
              "replay_assumption": "nominal hull/actuator model with observed future commands; future commands used only in offline dev calibration",
              "normalization_sha256": sha(args.normalization),
              "dev_manifest_sha256": sha(manifest_path),
              "dev_trace_sha256": input_hashes}
    args.out.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(result, indent=2) + "\n"
    output = args.out / "calibration.json"
    if output.exists() and output.read_text() != raw:
        raise RuntimeError("refusing to overwrite changed calibration")
    output.write_text(raw)
    run_record = {"schema": 1, "method": "h15_e1_dev_replay_calibration",
                  "normalization_sha256": sha(args.normalization),
                  "dev_collection_run_sha256": sha(args.collection / "run.json"),
                  "source_sha256": {p: sha(ROOT / p) for p in (
                      "experiments/h15_mainline/response_estimator.py",
                      "experiments/h15_mainline/calibrate_response_intervals.py")},
                  "calibration_sha256": sha(output),
                  "python_executable": "/bin/python3"}
    (args.out / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    (args.out / "console.log").write_text(
        f"calibrated {counts['1']} dev replay starts at horizons 1, 5, 25, 35\n")
    print(json.dumps({"counts": counts, "radii": radii,
                      "source_counts": dict(source_counts)}, indent=2))


if __name__ == "__main__":
    main()
