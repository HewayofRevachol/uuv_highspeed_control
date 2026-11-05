"""Run the 12 short E0 schema/compensation/environment contract probes."""
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from experiments.h15_cat_mpc.run_episode import ROOT, run_episode, sha256
from experiments.h15_cat_mpc.residual_predictor import RESIDUAL_LABEL_VERSION
from experiments.h15_cat_mpc.p0_probe import CASES, run_one as run_gate_control


DISTURBANCE_ORACLE = [0., 0., 0., .04, -.04]


def execute(job):
    torch.set_num_threads(1)
    return run_episode(**job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    out = args.out.resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    path = ROOT/"data/datasets/h15_mainline_quickstart/train_late_entry.csv"
    jobs = []
    for profile in ("matched_nominal", "known_disturbance", "target"):
        for compensation in ("on", "off"):
            for action in ("step", "reversal"):
                name = f"{profile}_{compensation}_{action}"
                jobs.append({"path": path, "seed": 71001,
                    "action_seed": 71001, "out": out/name, "method": "A0",
                    "target_domain": profile == "target",
                    "environment_profile": ("matched_nominal" if profile != "target" else "target"),
                    "compensation_mode": compensation, "action_profile": action,
                    "known_disturbance": (DISTURBANCE_ORACLE
                        if profile == "known_disturbance" else None),
                    "predictor_mode": "zero", "max_steps": 40})
    protocol = {"schema": 1, "stage": "E0 contract probe", "periods_per_job": 40,
        "jobs": 12, "seed": 71001, "geometry": str(path.relative_to(ROOT)),
        "profiles": ["matched_nominal", "known_disturbance", "target"],
        "compensation_modes": ["on", "off"], "actions": ["step", "reversal"],
        "oracle_disturbance": DISTURBANCE_ORACLE,
        "oracle_is_test_only": True, "holdout_used": False}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=mp.get_context("spawn")) as pool:
        results = list(pool.map(execute, jobs))
    rows, errors = [], []
    for job, result in zip(jobs, results):
        profile = ("matched_nominal" if job["out"].name.startswith("matched_nominal_")
            else "known_disturbance" if job["out"].name.startswith("known_disturbance_")
            else "target")
        if len(result["trace"]) > 40 or not result["trace"]:
            errors.append(f"invalid period count: {job['out'].name}")
        if result["command_schema"] != "issued_v2":
            errors.append(f"bad command schema: {job['out'].name}")
        if len(result["completed_rows"]) != len(result["trace"]):
            errors.append(f"terminal period/log count mismatch: {job['out'].name}")
        for period, (command_row, trace) in enumerate(zip(
                result["completed_rows"], result["trace"])):
            issued = np.asarray(json.loads(command_row["issued_command_json"]))
            compat = np.asarray(json.loads(command_row["command_json"]))
            if not np.allclose(issued, compat, atol=1e-6, rtol=0.) or not np.allclose(
                    issued, trace["command"], atol=1e-6, rtol=0.):
                errors.append(f"command mismatch: {job['out'].name}")
            if command_row.get("residual_label_version") != RESIDUAL_LABEL_VERSION:
                errors.append(f"label version mismatch: {job['out'].name}")
            if int(round(float(command_row["time_s"])/.1)) != period:
                errors.append(f"period alignment mismatch: {job['out'].name}")
            first_substep = period*5
            if first_substep < len(result["strict_substeps"]):
                queued = np.asarray(result["strict_substeps"][first_substep]["queue"])[0]
                if not np.allclose(issued, queued, atol=1e-6, rtol=0.):
                    errors.append(f"issued command differs from queue head: {job['out'].name}")
            if (json.loads(command_row["theta_label_before_json"]) !=
                    json.loads(command_row["theta_used_json"]) or
                    len(json.loads(command_row["alpha_used_json"])) != 2 or
                    len(command_row["d_forecast"]) != 35):
                errors.append(f"theta/alpha/forecast alignment mismatch: {job['out'].name}")
        if profile == "target":
            snapshot = result.get("target_domain_reset_snapshot")
            if not result.get("target_domain_applied_after_reset") or snapshot is None:
                errors.append(f"missing target reset snapshot: {job['out'].name}")
            else:
                before = np.asarray(snapshot["parameters_fin_before"])
                after = np.asarray(snapshot["parameters_fin_after"])
                if not np.allclose(after, before*.9, atol=1e-12, rtol=1e-10):
                    errors.append(f"target gain mismatch: {job['out'].name}")
        if profile == "matched_nominal":
            scale = np.asarray([1., 1., 1., .1, .1])
            residual = np.asarray([x["residual_body_acc"] for x in result["completed_rows"]])
            maximum = float(np.max(np.abs(residual/scale))) if len(residual) else float("inf")
            if maximum > 1e-6:
                errors.append(f"matched nominal residual {maximum:.3g}: {job['out'].name}")
        rows.append({"job": job["out"].name, "profile": profile,
            "compensation_mode": job["compensation_mode"], "action_profile": job["action_profile"],
            "periods": len(result["trace"]), "success": result["success"],
            "command_schema": result["command_schema"],
            "target_domain_applied_after_reset": result["target_domain_applied_after_reset"],
            "residual_label_version": result["residual_label_version"],
            "saturation_fraction": result["saturation_fraction"],
            "nominal_residual_max_scaled": (maximum if profile == "matched_nominal" else None),
            "fallback_periods": result["fallback_periods"],
            "episode_sha256": sha256(job["out"]/"episode.json")})
    base_control = run_gate_control(CASES["simple"], False, 40)
    gated_control = run_gate_control(CASES["simple"], True, 40)
    count = min(len(base_control), len(gated_control))
    command_delta = [float(np.linalg.norm(np.asarray(gated_control[i]["command"])-
        np.asarray(base_control[i]["command"]))) for i in range(count)]
    nonzero_gate_commands = sum(x > 1e-6 for x in command_delta)
    if nonzero_gate_commands == 0:
        errors.append("synthetic non-saturated 0/1 gate control did not change commands")
    report = {"schema": 1, "protocol": protocol, "jobs_completed": len(rows),
        "errors": errors, "passed": not errors, "results": rows,
        "synthetic_gate_control": {"base_cycles": len(base_control),
            "gated_cycles": len(gated_control),
            "command_delta_nonzero_cycles": nonzero_gate_commands,
            "max_command_delta_physical_units": max(command_delta, default=0.)},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), ROOT/"experiments/h15_cat_mpc/run_episode.py",
            ROOT/"experiments/h15_cat_mpc/residual_predictor.py",
            ROOT/"aav_mpc/h15_target_domain.py", ROOT/"simulator/batch_path_env.py",
            ROOT/"simulator/ddat_env.py")}}
    (out/"e0_report.json").write_text(json.dumps(report, indent=2)+"\n")
    if errors:
        raise RuntimeError("E0 checks failed: " + "; ".join(errors))
    print(json.dumps({"jobs_completed": len(rows), "passed": True,
                      "report": str((out/"e0_report.json").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
