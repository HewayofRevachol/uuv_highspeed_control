"""Measure the frozen online candidate screen on three Gate1 initial states.

This is a budget diagnostic, not a formal B0_shared episode or safety result.
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import time

import torch

from experiments.h15_mainline.candidate_filter import screen_candidate
from experiments.h15_mainline.estimator_io import load_frozen_estimator
from experiments.h15_mainline.progress_reference import ProgressReferenceEnv
from experiments.h15_mainline.run_smoke import CASES
from simulator.batch_path_env import PathBank
from simulator.model import LoopPath
from simulator.rti_symbolic import ActuatorAwareCpuDynamics


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    config = json.loads(args.config.read_text())
    if (config["candidate_grid_mps"] != freeze["candidate_grid_mps"] or
            config["candidate_limit_k"] != freeze["candidate_limit_k"] or
            config["screen_budget_ms"] != freeze["screen_budget_ms"] or
            config["estimator_checkpoint_sha256"] != sha(args.checkpoint)):
        raise RuntimeError("filter config differs from frozen inputs")
    checkpoint = json.loads(args.checkpoint.read_text())
    rows = []
    for label, split, case_id, seed, mismatch in CASES:
        item = freeze["manifests"][split]
        manifest_file = ROOT / item["path"]
        if sha(manifest_file) != item["sha256"]:
            raise RuntimeError(f"changed {split} manifest")
        manifest = json.loads(manifest_file.read_text())
        case = next(case for case in manifest["cases"] if case["id"] == case_id)
        path_file = ROOT / case["path"]
        if sha(path_file) != case["path_sha256"]:
            raise RuntimeError(f"changed path {path_file}")
        bank = PathBank([path_file], "cpu")
        bank.speed[:] = 15.
        bank.first_state[:, 5] = 15.
        bank.durations[:] = bank.lengths / 15.
        env = ProgressReferenceEnv(
            bank, batch_size=1, device="cpu", horizon=35,
            backend="rti_cpu", seed=seed, randomize_plant=mismatch,
            admm_iterations=400, admm_rho=10.,
            attitude_pitch_weight=100., attitude_yaw_weight=30.,
            fsp_enabled=False, preview_mode="multiscale", yaw_reward="course")
        env.mpc.dynamics = ActuatorAwareCpuDynamics(.1, fin_gain_scale=.85,
                                                    fin_tau_scale=2.)
        env.reset(torch.zeros(1, dtype=torch.long))
        if not mismatch:
            env.disturbance_std.zero_()
        estimator = load_frozen_estimator(args.checkpoint)
        estimate = estimator._fallback("short_history", time.perf_counter())
        path = LoopPath(path_file)
        candidates = [screen_candidate(env, path, candidate, estimate,
                                       config, checkpoint["parameters"])
                      for candidate in config["candidate_grid_mps"]]
        rows.append({"case_label": label, "case_id": case_id, "seed": seed,
                     "snapshot_period": 0,
                     "plant_scope": "randomized_reset" if mismatch else "nominal_reset",
                     "estimator_source": estimate.source,
                     "estimator_fallback_reason": estimate.fallback_reason,
                     "physical_candidates_mps": [item["candidate_mps"] for item in candidates
                                                 if item["physical_accepted"]],
                     "budget_accepted_mps": [item["candidate_mps"] for item in candidates
                                             if item["online_accepted"]],
                     "candidates": candidates})
    report = {"schema": 1, "scope": "Gate1 initial-state budget diagnostic only",
              "screen_budget_ms": config["screen_budget_ms"], "cases": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    result_file = args.out / "candidate_budget.json"
    raw = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if result_file.exists() and result_file.read_text() != raw:
        raise RuntimeError("refusing to replace different candidate budget result")
    result_file.write_text(raw)
    source_files = ("experiments/h15_mainline/probe_gate1_budget.py",
                    "experiments/h15_mainline/candidate_filter.py",
                    "experiments/h15_mainline/progress_reference.py")
    run = {"schema": 1, "method": "h15_gate1_initial_candidate_budget_probe",
           "freeze_sha256": sha(args.freeze), "checkpoint_sha256": sha(args.checkpoint),
           "filter_config_sha256": sha(args.config),
           "source_sha256": {name: sha(ROOT / name) for name in source_files},
           "result_sha256": sha(result_file), "python_executable": "/bin/python3",
           "python": platform.python_version(),
           "threads": {key: os.environ.get(key) for key in
                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    run_file = args.out / "run.json"
    run_raw = json.dumps(run, indent=2) + "\n"
    if run_file.exists() and run_file.read_text() != run_raw:
        raise RuntimeError("refusing to replace different run record")
    run_file.write_text(run_raw)
    summary = {"cases": len(rows),
               "budget_empty_cases": sum(not row["budget_accepted_mps"] for row in rows),
               "physical_empty_cases": sum(not row["physical_candidates_mps"] for row in rows),
               "all_candidates_over_budget": all(
                   item["total_ms"] > config["screen_budget_ms"]
                   for row in rows for item in row["candidates"])}
    (args.out / "console.log").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
