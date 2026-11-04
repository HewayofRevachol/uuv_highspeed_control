"""Recreate coupled abort states and inspect all candidate scenario rejections."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from experiments.h15_mainline.candidate_filter import screen_candidate, select_b2
from experiments.h15_mainline.estimator_io import load_frozen_estimator
from experiments.h15_mainline.run_offline_governor_smoke import ROOT, create_env
from simulator.model import LoopPath


HORIZONS = (1, 5, 25, 35)
FIELDS = ("lateral_m", "depth_m", "pitch_deg", "tangent_water_speed_mps")


def envelope_radii(probe):
    previous = dict.fromkeys(FIELDS, 0.)
    radii = {}
    for horizon in HORIZONS:
        values = probe["statistics"][str(horizon)]
        previous = {field: max(previous[field], values[field]["outside_q99"])
                    for field in FIELDS}
        radii[str(horizon)] = previous
    return radii


def compact(candidate):
    return {"candidate_mps": candidate["candidate_mps"],
            "cost": candidate.get("cost"),
            "qp_status": candidate["qp_status"],
            "physical_accepted": candidate["physical_accepted"],
            "first_rejection": candidate["first_rejection"],
            "scenarios": [{"name": scenario["scenario"],
                           "first_rejection": scenario["first_rejection"],
                           "first_time_s": scenario["first_time_s"],
                           "max_abs_pitch_deg": scenario["maxima"]["pitch_deg"],
                           "min_pitch_margin_deg": scenario["min_margins"]["pitch_deg"]}
                          for scenario in candidate["scenarios"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("B0_shared", "B2_gov"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-period", type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    target_period = args.target_period if args.target_period is not None else (
        13 if args.method == "B0_shared" else 15)
    path_file = ROOT / "data/datasets/h15_mainline_quickstart/dev_coupled.csv"
    path = LoopPath(path_file)
    env = create_env(path_file, 32001, False)
    config = json.loads(args.config.read_text())
    checkpoint = json.loads(args.checkpoint.read_text())
    estimator = load_frozen_estimator(args.checkpoint)
    radii = envelope_radii(json.loads(args.envelope.read_text()))
    prior_run = ROOT / "outputs/experiments/gov_uuv_h15_smoke_s32001" / (
        "full_b0_coupled_20261104" if args.method == "B0_shared" else
        "full_b2_coupled_20261104")
    prior_cycles = [json.loads(line) for line in
                    (prior_run / "cycle_log.jsonl").read_text().splitlines()]
    selected_history = []
    for period in range(target_period + 1):
        expected = np.asarray(prior_cycles[period]["state_before"])
        if not np.allclose(env.state[0].numpy(), expected, rtol=0., atol=1e-8):
            difference = float(np.max(np.abs(env.state[0].numpy()-expected)))
            raise RuntimeError(f"state reconstruction diverged at period {period}: "
                               f"max_abs={difference}, selections={selected_history}, "
                               f"recorded={[row['selected_candidate_mps'] for row in prior_cycles[:period]]}")
        estimate = estimator._fallback("gate1_e0_short_history", 0.)
        original = [screen_candidate(env, path, candidate, estimate,
                                     config, checkpoint["parameters"])
                    for candidate in config["candidate_grid_mps"]]
        if period == target_period:
            envelope_estimate = replace(estimate, constraint_radii=radii)
            changed = [screen_candidate(env, path, candidate, envelope_estimate,
                                        config, checkpoint["parameters"])
                       for candidate in config["candidate_grid_mps"]]
            result = {"schema": 1, "scope": "dev-only diagnostic, not formal screening",
                      "method": args.method, "period": period,
                      "progress_m": float(env.progress[0]),
                      "original_radii": estimate.constraint_radii,
                      "envelope_outside_radii": radii,
                      "original": [compact(item) for item in original],
                      "envelope_outside": [compact(item) for item in changed]}
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"method": args.method, "period": period,
                              "original_accepted": [item["candidate_mps"]
                                                    for item in original if item["physical_accepted"]],
                              "envelope_accepted": [item["candidate_mps"]
                                                    for item in changed if item["physical_accepted"]]},
                             indent=2))
            return
        accepted = [item["candidate_mps"] for item in original if item["physical_accepted"]]
        selected = (15. if 15. in accepted else None) if args.method == "B0_shared" else \
                   select_b2(original, online=False)
        if selected is None:
            raise RuntimeError(f"early candidate failure at period {period}")
        selected_history.append(selected)
        env.set_intent(selected)
        env.defer_mpc_solve = False
        env.base_command, env.last_mpc_ok = env._solve_mpc()
        env.defer_mpc_solve = True
        if not bool(env.last_mpc_ok[0]):
            raise RuntimeError(f"early QP failure at period {period}")
        env.step_verified_command()


if __name__ == "__main__":
    main()
