"""Lower-bound timing for five candidate-specific N35 solves on train states."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.h15_mainline.progress_reference import ProgressReferenceEnv
from simulator.batch_path_env import PathBank
from simulator.rti_symbolic import ActuatorAwareCpuDynamics


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    manifest_path = ROOT / freeze["manifests"]["train"]["path"]
    if sha(manifest_path) != freeze["manifests"]["train"]["sha256"]:
        raise RuntimeError("train manifest changed")
    manifest = json.loads(manifest_path.read_text())
    grid = freeze["candidate_grid_mps"]
    if len(grid) != freeze["candidate_limit_k"]:
        raise RuntimeError("grid exceeds frozen K")
    results = []
    for case in manifest["cases"]:
        path = ROOT / case["path"]
        if sha(path) != case["path_sha256"]:
            raise RuntimeError(f"train path changed: {path}")
        bank = PathBank([path], "cpu")
        bank.speed[:] = 15.
        bank.first_state[:, 5] = 15.
        bank.durations[:] = bank.lengths / 15.
        env = ProgressReferenceEnv(bank, batch_size=1, device="cpu", horizon=35,
                                   backend="rti_cpu", seed=31001,
                                   randomize_plant=False, current_max_mps=0.,
                                   admm_iterations=400, admm_rho=10.,
                                   attitude_pitch_weight=100., attitude_yaw_weight=30.,
                                   fsp_enabled=False, preview_mode="multiscale",
                                   yaw_reward="course")
        env.mpc.dynamics = ActuatorAwareCpuDynamics(.1, fin_gain_scale=.85,
                                                    fin_tau_scale=2.)
        env.reset(torch.zeros(1, dtype=torch.long))
        env.set_intent(15.)
        for target_step in (0, 50, 100):
            while int(env.steps[0]) < target_step:
                _, _, done, _ = env.step(torch.zeros(1, 3))
                if bool(done[0]):
                    raise RuntimeError("train episode ended before benchmark state")
            state = env.observed_augmented_state()
            plan = env.mpc.plan.clone()
            previous = env.mpc.previous.clone()
            states = env.mpc.states.clone()
            candidate_rows = []
            started = time.perf_counter()
            for candidate in grid:
                env.set_intent(candidate)
                reference = env._reference()
                tic = time.perf_counter()
                _, valid = env.mpc.solve(state, reference)
                elapsed_ms = (time.perf_counter()-tic)*1000.
                candidate_rows.append({"candidate_mps": candidate,
                                       "qp_valid": bool(valid[0]),
                                       "qp_status": env.mpc.last_status[0],
                                       "solve_ms": elapsed_ms})
                env.mpc.plan[:] = plan
                env.mpc.previous[:] = previous
                env.mpc.states[:] = states
            total_ms = (time.perf_counter()-started)*1000.
            results.append({"case_id": case["id"], "step": target_step,
                            "candidate_solves": candidate_rows,
                            "five_solve_wall_ms": total_ms,
                            "screen_budget_ms": freeze["screen_budget_ms"],
                            "budget_exceeded": total_ms > freeze["screen_budget_ms"]})
            env.set_intent(15.)
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "candidate_solve_benchmark.json"
    result_path.write_text(json.dumps({"schema": 1,
                                       "scope": "QP solve lower bound; no nonlinear replay or state feasibility",
                                       "states": results}, indent=2) + "\n")
    totals = np.asarray([item["five_solve_wall_ms"] for item in results])
    summary = {"states": len(results), "minimum_five_solve_ms": float(totals.min()),
               "median_five_solve_ms": float(np.median(totals)),
               "maximum_five_solve_ms": float(totals.max()),
               "budget_exceeded_count": sum(item["budget_exceeded"] for item in results)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    run_record = {"schema": 1, "method": "h15_n35_five_candidate_solve_lower_bound",
                  "freeze_sha256": sha(args.freeze), "train_manifest_sha256": sha(manifest_path),
                  "source_sha256": {p: sha(ROOT / p) for p in (
                      "experiments/h15_mainline/benchmark_candidate_solves.py",
                      "experiments/h15_mainline/progress_reference.py",
                      "aav_mpc/condensed_mpc.py")},
                  "python_executable": "/bin/python3", "python": platform.python_version(),
                  "threads": {key: os.environ.get(key) for key in
                              ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                  "result_sha256": sha(result_path)}
    (args.out / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    (args.out / "console.log").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
