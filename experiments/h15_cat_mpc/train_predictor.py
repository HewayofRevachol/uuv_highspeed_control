"""Generate causal residual labels and fit the train-only AR(1) model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.h15_cat_mpc.residual_predictor import (ResidualAR1,
    nominal_delay_tau, nominal_period_rollout)
from experiments.h15_cat_mpc.adapter import THETA0
from experiments.h15_mainline.response_estimator import SlidingWindowResponseEstimator
from simulator.batch_mpc import BatchedUUVModel
from simulator.model import MISMATCHED_PLANT


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_array(row, key):
    return np.asarray(json.loads(row[key]), dtype=float)


def residual_for(model, row, theta, nominal_rate):
    before = torch.as_tensor(read_array(row, "state_before_json"), dtype=torch.float64)[None]
    after = read_array(row, "state_after_json")
    actual = torch.as_tensor(read_array(row, "actuator_before_json"), dtype=torch.float64)[None]
    queue = torch.as_tensor(read_array(row, "queue_before_json"), dtype=torch.float64)[None]
    issued = torch.as_tensor(read_array(row, "issued_command_json"), dtype=torch.float64)[None]
    delay, tau = nominal_delay_tau(fin_tau_scale=theta[1])
    rate = nominal_rate
    p = MISMATCHED_PLANT
    parameters = torch.tensor([[p.mass_scale, p.damping_scale, p.thrust_gain,
        p.elevator_gain*theta[0], p.rudder_gain*theta[0]]], dtype=torch.float64)
    predicted, _, next_rate, _ = nominal_period_rollout(model, before, actual, rate, queue, delay,
        tau, issued, parameters,
        second_order=row.get("second_order_actuator", "True").lower() == "true",
        fin_tau_scale=float(theta[1]),
        integrator=row.get("integrator", "True").lower() == "true")
    value = (after[5:10]-predicted[0, 5:10].numpy())/.1
    return value, next_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compensation-mode", choices=("on", "off"), default="on")
    args = parser.parse_args()
    all_summaries = json.loads((args.collection/"summary.json").read_text())
    summaries = [row for row in all_summaries
                 if row.get("compensation_mode", "on") == args.compensation_mode]
    expected = 12
    if len(summaries) != expected or len({(x["case_id"], x["seed"]) for x in summaries}) != expected:
        raise RuntimeError("predictor training requires the 12 unique train episodes")
    normalization = json.loads(args.normalization.read_text())
    if (normalization.get("split") != "train" or
            normalization.get("compensation_mode") != args.compensation_mode):
        raise RuntimeError("normalization must be train-only and compensation-mode matched")
    model = BatchedUUVModel("cpu", torch.float64)
    residuals, episode_residuals, labels_manifest = [], [], {}
    args.out.mkdir(parents=True, exist_ok=False)
    label_path = args.out/"residual_labels.csv"
    with label_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case_id", "seed", "step", "theta_label_gain",
                         "theta_label_tau_scale", "r_u", "r_v", "r_w", "r_q", "r_r"))
        for summary in summaries:
            trace_path = ROOT/summary["trace"]
            if sha(trace_path) != summary["trace_sha256"]:
                raise RuntimeError(f"training trace hash changed: {trace_path}")
            estimator = SlidingWindowResponseEstimator(normalization)
            theta = np.asarray(THETA0, dtype=float)
            nominal_rate = torch.zeros((1, 3), dtype=torch.float64)
            current_episode = []
            with trace_path.open(newline="") as trace:
                for row in csv.DictReader(trace):
                    if (row.get("command_schema") != "issued_v2" or
                            row.get("residual_label_version") != "nominal_5x20ms_issued_v2"):
                        raise RuntimeError("refusing legacy/base_v1 or mixed residual training trace")
                    theta_before = theta.copy()
                    r, nominal_rate = residual_for(model, row, theta_before,
                        nominal_rate)
                    residuals.append(r)
                    current_episode.append(r)
                    writer.writerow((summary["case_id"], summary["seed"], row["step"],
                                     *theta_before, *r))
                    estimate = estimator.update(row)
                    theta = np.asarray([estimate.fin_gain, estimate.fin_tau_scale])
            episode_residuals.append(np.asarray(current_episode))
            labels_manifest[str(trace_path.relative_to(ROOT))] = summary["trace_sha256"]
    values = np.asarray(residuals, dtype=float)
    predictor = ResidualAR1.fit_episodes(episode_residuals)
    predictor.version = "train_ar1_issued_v2"
    predictor.save(args.out/"predictor.json", {
        "split": "train", "compensation_mode": args.compensation_mode,
        "episodes": len(summaries), "labels": len(values),
        "label_version": "nominal_5x20ms_issued_v2",
        "label_contract": {"state_channels": [5, 6, 7, 8, 9],
            "period_s": .1, "substep_s": .02, "substeps": 5,
            "theta_source": "theta_label_before_json; causal estimate available before period",
            "actuator_model": "fixed nominal delay/tau; causal second-order rate state",
            "hidden_plant_parameters_used": False,
            "formula": "(state_after[5:10]-nominal_state_after[5:10])/.1"},
        "normalization_sha256": sha(args.normalization),
        "trace_sha256": labels_manifest,
        "source_sha256": {"experiments/h15_cat_mpc/train_predictor.py": sha(__file__),
                          "experiments/h15_mainline/response_estimator.py": sha(
                              ROOT/"experiments/h15_mainline/response_estimator.py")}})
    one_step_errors, persistence_errors = [], []
    for sequence in episode_residuals:
        one_step_prediction = predictor.mean + predictor.phi*(sequence[:-1]-predictor.mean)
        one_step_errors.extend(np.abs(sequence[1:]-one_step_prediction))
        persistence_errors.extend(np.abs(sequence[1:]-sequence[:-1]))
    metrics = {"episodes": len(summaries), "residual_labels": len(values),
        "residual_mean": values.mean(0).tolist(), "residual_std": values.std(0).tolist(),
        "persistence_one_step_mae": np.mean(persistence_errors, axis=0).tolist(),
        "ar1_one_step_mae": np.mean(one_step_errors, axis=0).tolist(),
        "scope": "train only; dev evaluation remains separate"}
    (args.out/"training_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
