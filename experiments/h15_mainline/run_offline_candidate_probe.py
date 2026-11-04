"""Probe candidate-specific QP plus nonlinear screen on frozen train states."""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.h15_mainline.candidate_filter import screen_candidate, select_b2
from experiments.h15_mainline.estimator_io import load_frozen_estimator
from experiments.h15_mainline.progress_reference import ProgressReferenceEnv
from simulator.batch_path_env import PathBank
from simulator.model import LoopPath
from simulator.rti_symbolic import ActuatorAwareCpuDynamics


ROOT = Path(__file__).resolve().parents[2]
STEPS = (10, 50, 100)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    checkpoint = json.loads(args.checkpoint.read_text())
    config = json.loads(args.config.read_text())
    if config["estimator_checkpoint_sha256"] != sha(args.checkpoint):
        raise RuntimeError("estimator checkpoint differs from filter config")
    if (config["candidate_grid_mps"] != freeze["candidate_grid_mps"] or
            config["candidate_limit_k"] != freeze["candidate_limit_k"] or
            config["screen_budget_ms"] != freeze["screen_budget_ms"]):
        raise RuntimeError("candidate config differs from frozen quickstart")
    manifest_path = ROOT / freeze["manifests"]["train"]["path"]
    if sha(manifest_path) != freeze["manifests"]["train"]["sha256"]:
        raise RuntimeError("train manifest changed")
    manifest = json.loads(manifest_path.read_text())
    collection_run = json.loads((args.train_collection / "run.json").read_text())
    if collection_run["train_manifest_sha256"] != sha(manifest_path):
        raise RuntimeError("train observations use another manifest")
    collection_summaries = json.loads((args.train_collection / "summary.json").read_text())
    results = []
    for case in manifest["cases"]:
        path_file = ROOT / case["path"]
        if sha(path_file) != case["path_sha256"]:
            raise RuntimeError(f"train path changed: {path_file}")
        trace_item = next(item for item in collection_summaries
                          if item["case_id"] == case["id"] and item["seed"] == 31001)
        trace_file = ROOT / trace_item["trace"]
        if sha(trace_file) != trace_item["trace_sha256"]:
            raise RuntimeError(f"train observation trace changed: {trace_file}")
        with trace_file.open(newline="") as stream:
            trace_rows = list(csv.DictReader(stream))
        path = LoopPath(path_file)
        bank = PathBank([path_file], "cpu")
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
        env.disturbance_std.zero_()
        estimator = load_frozen_estimator(args.checkpoint)
        latest = None
        for target_step in STEPS:
            while int(env.steps[0]) < target_step:
                _, _, done, _ = env.step(torch.zeros(1, 3))
                row = trace_rows[int(env.steps[0])-1]
                observed = np.asarray(json.loads(row["state_after_json"]))
                if np.max(np.abs(observed-env.state[0].numpy())) > 1e-8:
                    raise RuntimeError("live train state differs from frozen trace")
                latest = estimator.update(row)
                if bool(done[0]):
                    raise RuntimeError("episode ended before probe state")
            if latest is None:
                raise RuntimeError("estimator history unavailable")
            original_dynamics = env.mpc.dynamics
            tic = time.perf_counter()
            env.mpc.dynamics = ActuatorAwareCpuDynamics(
                .1, fin_gain_scale=latest.fin_gain,
                fin_tau_scale=latest.fin_tau_scale)
            model_update_ms = (time.perf_counter()-tic)*1000.
            try:
                candidates = [screen_candidate(env, path, candidate, latest,
                                               config, checkpoint["parameters"])
                              for candidate in config["candidate_grid_mps"]]
            finally:
                env.mpc.dynamics = original_dynamics
            offline_set = [item["candidate_mps"] for item in candidates
                           if item["physical_accepted"]]
            online_set = [item["candidate_mps"] for item in candidates
                          if item["online_accepted"]]
            results.append({"case_id": case["id"], "seed": 31001,
                            "step": target_step, "estimator_source": latest.source,
                            "estimator_fallback_reason": latest.fallback_reason,
                            "estimator_gain": latest.fin_gain,
                            "estimator_tau_scale": latest.fin_tau_scale,
                            "model_update_ms": model_update_ms,
                            "offline_empirical_candidates": offline_set,
                            "online_budgeted_candidates": online_set,
                            "offline_b2_choice": select_b2(candidates),
                            "online_b2_choice": select_b2(candidates, online=True),
                            "b0_15_offline_accepted": 15. in offline_set,
                            "candidate_results": candidates})
            print(f"{case['id']} step {target_step}: offline={offline_set} online={online_set}",
                  flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "candidate_probe.json"
    result_path.write_text(json.dumps({"schema": 1,
                                       "scope": "train-state offline empirical mechanism probe",
                                       "states": results}, indent=2) + "\n")
    summary = {"states": len(results),
               "offline_candidate_empty": sum(not item["offline_empirical_candidates"] for item in results),
               "online_candidate_empty": sum(not item["online_budgeted_candidates"] for item in results),
               "b0_15_offline_accepted": sum(item["b0_15_offline_accepted"] for item in results),
               "b2_offline_selected": sum(item["offline_b2_choice"] is not None for item in results),
               "candidate_reasons": {reason: sum(
                   candidate["first_rejection"] == reason for state in results
                   for candidate in state["candidate_results"])
                   for reason in sorted({candidate["first_rejection"]
                                         for state in results
                                         for candidate in state["candidate_results"]
                                         if candidate["first_rejection"]})}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    run_record = {"schema": 1, "method": "h15_offline_candidate_specific_n35_probe",
                  "freeze_sha256": sha(args.freeze), "checkpoint_sha256": sha(args.checkpoint),
                  "filter_config_sha256": sha(args.config),
                  "train_collection_run_sha256": sha(args.train_collection / "run.json"),
                  "source_sha256": {p: sha(ROOT / p) for p in (
                      "experiments/h15_mainline/run_offline_candidate_probe.py",
                      "experiments/h15_mainline/candidate_filter.py",
                      "experiments/h15_mainline/progress_reference.py")},
                  "python_executable": "/bin/python3", "python": platform.python_version(),
                  "threads": {key: os.environ.get(key) for key in
                              ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                  "result_sha256": sha(result_path)}
    (args.out / "run.json").write_text(json.dumps(run_record, indent=2) + "\n")
    (args.out / "console.log").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
