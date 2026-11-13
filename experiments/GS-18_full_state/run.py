"""Paired GS-18 GPU ADMM comparison on ten hull states."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.gs12_rl_beta_gpu_admm.compare_mpc_rl import load_policy, read_jsonl
from experiments.gs12_joint_route.joint_reference import joint_task_reference

from metrics import DT, N_STEPS, NAMES, full_reference, score
from traced_env import TracedBetaEnv


BETAS = (0., .5, .55, .6, .65, .75)
METHODS = tuple(f"beta_{beta:g}" for beta in BETAS) + ("RL_beta",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def evaluate(method, tasks, policy, settings, batch_size, trace_dir,
             fixed_initial_beta=None, attitude_scale=1., velocity_scale=1., position_scale=1.):
    rows = []
    for start in range(0, len(tasks), batch_size):
        group = tasks[start:start+batch_size]
        beta = float(method.removeprefix("beta_")) if method != "RL_beta" else .75
        initial_beta = fixed_initial_beta if fixed_initial_beta is not None and method != "RL_beta" else beta
        env = TracedBetaEnv(group, [1.]*len(group), initial_beta=initial_beta,
                            predictor=settings["predictor"],
                            admm_iterations=settings["iterations"],
                            admm_rho=settings["rho"],
                            admm_tolerance=settings["tolerance"],
                            cuda_graph=settings["cuda_graph"])
        env.mpc.state_weight[3:5] *= attitude_scale
        env.mpc.state_weight[5:8] *= velocity_scale
        env.mpc.state_weight[:3] *= position_scale
        targets = [[] for _ in group]
        began = time.perf_counter()
        for _ in range(env.decisions):
            if not bool(env.alive.any()):
                break
            target = (policy.deterministic(env.observe()).to(torch.float64)
                      if method == "RL_beta" else torch.full_like(env.beta, beta))
            for i, active in enumerate(env.alive.cpu().tolist()):
                if active:
                    targets[i].append(float(target[i]))
            env.step_target(target)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-began
        states = env.trace_x.cpu().numpy()
        actuals = env.trace_actual.cpu().numpy()
        valids = env.trace_valid.cpu().numpy()
        commands = env.trace_command.cpu().numpy()
        betas = env.trace_beta.cpu().numpy()
        qp_valids = env.trace_qp_valid.cpu().numpy()
        residuals = env.trace_qp_residual.cpu().numpy()
        for i, (task, base) in enumerate(zip(group, env.summary())):
            n = int(base["control_cycles"])*5
            reference = full_reference(task)
            measured = score(states[i], reference, base["status"])
            if measured["J_track_recomputed"] is not None:
                np.testing.assert_allclose(measured["J_track_recomputed"], base["J_track"],
                                           rtol=1e-6, atol=1e-12)
            artifact = f"{method}__{task['task_id']}.npz"
            np.savez_compressed(trace_dir/artifact, time_s=np.arange(n+1)*DT,
                                x=states[i, :n+1], reference=reference[:n+1],
                                actual=actuals[i, :n+1], valid=valids[i, :n+1],
                                command=commands[i, :n//5], beta=betas[i, :n//5],
                                qp_valid=qp_valids[i, :n//5],
                                qp_residual=residuals[i, :n//5])
            scale = env.dynamics.scale.cpu().numpy()
            command_norm = commands[i, :n//5]/scale
            actual_norm = actuals[i, :n+1]/scale
            row = {"task_id": task["task_id"], "method": method,
                   "status": base["status"], "control_cycles": base["control_cycles"],
                   "J_track": base["J_track"], **measured,
                   "command_saturation_cycles": int(np.any(
                       (command_norm >= 1.-1e-6) |
                       (command_norm <= np.array([0., -1., -1.])+1e-6), axis=1).sum()),
                   "actual_saturation_steps": int(np.any(
                       (actual_norm >= 1.-1e-6) |
                       (actual_norm <= np.array([0., -1., -1.])+1e-6), axis=1).sum()),
                   "max_command_slew_normalized": float(np.abs(np.diff(command_norm, axis=0)).max())
                   if n > 5 else 0.,
                   "max_qp_primal": float(residuals[i, :n//5, 0].max()),
                   "max_qp_dual": float(residuals[i, :n//5, 1].max()),
                   "qp_failures": int((~qp_valids[i, :n//5]).sum()),
                   "beta_targets": targets[i], "trace": f"traces/{artifact}",
                   "batch_wall_s": elapsed}
            rows.append(row)
        print(f"{method}: {min(start+batch_size, len(tasks))}/{len(tasks)}", flush=True)
    return rows


def summarize(rows, methods):
    summary = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        complete = [row for row in subset if row["J10"] is not None]
        summary[method] = {"tasks": len(subset), "complete": len(complete),
                           "failures": len(subset)-len(complete),
                           "mean_J10_complete_only": float(np.mean([row["J10"] for row in complete]))
                           if complete else None,
                           "mean_J_track_complete_only": float(np.mean([row["J_track"] for row in complete]))
                           if complete else None,
                           "status_counts": {status: sum(row["status"] == status for row in subset)
                                             for status in sorted({row["status"] for row in subset})}}
    eligible = [method for method in methods if method != "RL_beta" and
                summary[method]["complete"] == summary[method]["tasks"]]
    best = min(eligible, key=lambda method: summary[method]["mean_J10_complete_only"]) if eligible else None
    paired = None
    if best and "RL_beta" in summary and summary["RL_beta"]["complete"] == summary["RL_beta"]["tasks"]:
        baseline = {row["task_id"]: row for row in rows if row["method"] == best}
        rl = [row for row in rows if row["method"] == "RL_beta"]
        paired = {"baseline": best, "wins": sum(row["J10"] < baseline[row["task_id"]]["J10"] for row in rl),
                  "mean_change_fraction": summary["RL_beta"]["mean_J10_complete_only"] /
                  summary[best]["mean_J10_complete_only"]-1.}
    return {"methods": summary, "best_complete_fixed_beta": best, "RL_vs_best_fixed": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path(__file__).with_name("tasks_dev.jsonl"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/gs12_gpu_admm_20261112T051540Z/formal_1h/final_actor.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--fixed-initial-beta", type=float, default=None,
                        help="start fixed-beta runs here, then slew toward their target")
    parser.add_argument("--attitude-scale", type=float, default=1.)
    parser.add_argument("--velocity-scale", type=float, default=1.)
    parser.add_argument("--position-scale", type=float, default=1.)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    tasks = read_jsonl(args.tasks)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        parser.error("task IDs must be unique")
    run_file = args.checkpoint.parent/"run.json"
    training_run = json.loads(run_file.read_text(encoding="utf-8"))
    settings = {"predictor": training_run["predictor"], **training_run["admm"]}
    torch.set_num_threads(1)
    policy = load_policy(args.checkpoint)
    trace_dir = args.out/"traces"
    trace_dir.mkdir(parents=True)
    rows = [row for method in args.methods for row in evaluate(
        method, tasks, policy, settings, args.batch_size, trace_dir,
        args.fixed_initial_beta, args.attitude_scale, args.velocity_scale, args.position_scale)]
    with (args.out/"episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    result = summarize(rows, args.methods)
    result["provenance"] = {"tasks": str(args.tasks), "tasks_sha256": sha256(args.tasks),
                            "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
                            "training_run_sha256": sha256(run_file), "settings": settings,
                            "state_order": NAMES, "dt_s": DT,
                            "score_window_s": [DT, 64.], "batch_size": args.batch_size}
    result["provenance"]["fixed_initial_beta"] = args.fixed_initial_beta
    result["provenance"]["attitude_scale"] = args.attitude_scale
    result["provenance"]["velocity_scale"] = args.velocity_scale
    result["provenance"]["position_scale"] = args.position_scale
    (args.out/"summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n",
                                          encoding="utf-8")
    print(json.dumps(result["methods"], indent=2), flush=True)


if __name__ == "__main__":
    main()

