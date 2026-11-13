"""Compare a phase-aware beta schedule, fixed beta, and the frozen GS-12 policy."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import GROUPS, SCALES, full_reference
from traced_env import TracedBetaEnv
from experiments.gs12_rl_beta_gpu_admm.compare_mpc_rl import load_policy


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "outputs/gs12_gpu_admm_20261112T051540Z/formal_1h/final_actor.pt"
OUT = ROOT / "outputs/gs18_dynamic_beta_switch_v4"
TASKS = Path(__file__).with_name("tasks_dynamic_beta_switch_v3.jsonl")
BETAS = (0., .25, .5, .525, .55, .575, .6, .65, .75)


def read_tasks():
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("task IDs must be unique")
    return tasks


def input_hashes():
    paths = [TASKS, CHECKPOINT, CHECKPOINT.parent/"run.json", Path(__file__).resolve(),
             Path(__file__).with_name("metrics.py"), Path(__file__).with_name("traced_env.py"),
             ROOT/"experiments/gs12_joint_route/joint_reference.py",
             ROOT/"experiments/gs12_rl_beta_gpu_admm/environment.py",
             ROOT/"experiments/gs12_rl_beta_gpu_admm/dynamics.py",
             ROOT/"experiments/gs12_rl_beta_gpu_admm/mpc.py"]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def j10(x, reference, mask):
    error = x[mask]-reference[mask]
    error[:, 4] = np.arctan2(np.sin(error[:, 4]), np.cos(error[:, 4]))
    groups = {name: float(((error[:, part]/SCALES[part])**2).mean())
              for name, (part, _) in GROUPS.items()}
    return float(sum(weight*groups[name] for name, (_, weight) in GROUPS.items()))


@torch.no_grad()
def evaluate(tasks, method, policy, settings, trace_dir):
    beta = float(method.removeprefix("beta_")) if method.startswith("beta_") else .75
    env = TracedBetaEnv(tasks, [1.]*len(tasks), initial_beta=.75,
                        predictor=settings["predictor"],
                        admm_iterations=settings["iterations"],
                        admm_rho=settings["rho"], admm_tolerance=settings["tolerance"],
                        cuda_graph=settings["cuda_graph"])
    targets = [[] for _ in tasks]
    for _ in range(env.decisions):
        if not bool(env.alive.any()):
            break
        time_s = env.k*.1
        if method == "RL_beta":
            target = policy.deterministic(env.observe()).to(torch.float64)
        elif method == "schedule_beta":
            high_phase = [((time_s < 16. or time_s >= 40.) if task["order"] == "high_low_high"
                           else (16. <= time_s < 40.)) for task in tasks]
            target = torch.as_tensor([.25 if high else .75 for high in high_phase],
                                     device=env.device, dtype=env.beta.dtype)
        else:
            target = torch.full_like(env.beta, beta)
        for i in range(len(tasks)):
            targets[i].append(float(target[i]))
        env.step_target(target)
    torch.cuda.synchronize()
    rows = []
    for i, (task, base) in enumerate(zip(tasks, env.summary())):
        n = int(base["control_cycles"])*5
        x = env.trace_x[i, :n+1].cpu().numpy()
        reference = full_reference(task)[:n+1]
        t = np.arange(n+1)*.02
        all_mask = (t >= .02) & (t <= 64.)
        middle_mask = (t >= 22.) & (t < 42.)
        ends_mask = ((t >= .02) & (t < 18.)) | ((t >= 46.) & (t <= 64.))
        high_mask, low_mask = ((ends_mask, middle_mask) if task["order"] == "high_low_high"
                               else (middle_mask, ends_mask))
        beta_trace = env.trace_beta[i, :n//5].cpu().numpy()
        qp = env.trace_qp_valid[i, :n//5].cpu().numpy()
        command = env.trace_command[i, :n//5].cpu().numpy()
        actual = env.trace_actual[i, :n+1].cpu().numpy()
        scale = env.dynamics.scale.cpu().numpy()
        command_norm, actual_norm = command/scale, actual/scale
        command_low = np.array([0., -1., -1.])
        row = {"task_id": task["task_id"], "method": method, "status": base["status"],
               "control_cycles": base["control_cycles"], "J10": j10(x, reference, all_mask),
               "J10_low_stages": j10(x, reference, low_mask),
               "J10_high_stage": j10(x, reference, high_mask),
               "qp_failures": int((~qp).sum()), "beta_targets": targets[i],
               "command_saturation_cycles": int(np.any((command_norm >= 1.-1e-6) |
                                                         (command_norm <= command_low+1e-6), axis=1).sum()),
               "actual_saturation_steps": int(np.any((actual_norm >= 1.-1e-6) |
                                                       (actual_norm <= command_low+1e-6), axis=1).sum()),
               "beta_min": float(beta_trace.min()), "beta_max": float(beta_trace.max()),
               "beta_mean": float(beta_trace.mean())}
        name = f"{method}__{task['task_id']}.npz"
        np.savez_compressed(trace_dir/name, time_s=t, x=x, reference=reference,
                            beta=beta_trace, qp_valid=qp)
        row["trace"] = f"traces/{name}"
        rows.append(row)
    return rows


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError(f"output directory must be empty: {OUT}")
    run = json.loads((CHECKPOINT.parent/"run.json").read_text(encoding="utf-8"))
    settings = {"predictor": run["predictor"], **run["admm"]}
    tasks = read_tasks()
    policy = load_policy(CHECKPOINT)
    methods = [*(f"beta_{beta:g}" for beta in BETAS), "RL_beta", "schedule_beta"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"traces").mkdir()
    (OUT/"protocol.json").write_text(json.dumps({
        "tasks": str(TASKS.resolve()), "checkpoint": str(CHECKPOINT.resolve()),
        "settings": settings, "methods": methods, "initial_beta": .75,
        "beta_targets": {"low_frequency": .75, "high_frequency": .25},
        "target_switch_times_s": [16., 40.],
        "reference_transition_centers_s": [20., 44.],
        "beta_slew_limit_per_0.1s": .02,
        "input_sha256": input_hashes(),
        "score": "unchanged J10; transition samples excluded only from phase diagnostics",
        "evidence": "exploratory, selected after the frequency screen; not independent validation"
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT/"tasks.jsonl").write_bytes(TASKS.read_bytes())
    rows = []
    for method in methods:
        rows.extend(evaluate(tasks, method, policy, settings, OUT/"traces"))
        print(f"{method}: {sum(row['method']==method for row in rows)}/{len(tasks)}", flush=True)
    (OUT/"episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    summary = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        completed = sum(row["status"].startswith("complete") for row in subset)
        summary[method] = {"completed": completed,
                           "tasks": len(subset), "qp_failures": sum(row["qp_failures"] for row in subset),
                           "command_saturation_cycles": sum(row["command_saturation_cycles"] for row in subset),
                           "actual_saturation_steps": sum(row["actual_saturation_steps"] for row in subset),
                           "eligible": completed == len(subset) and not any(
                               row["qp_failures"] or row["command_saturation_cycles"] or
                               row["actual_saturation_steps"] for row in subset),
                           "mean_J10": float(np.mean([row["J10"] for row in subset])),
                           "mean_J10_low_stages": float(np.mean([row["J10_low_stages"] for row in subset])),
                           "mean_J10_high_stage": float(np.mean([row["J10_high_stage"] for row in subset]))}
    fixed = [method for method in methods if method.startswith("beta_")]
    best_observed = min(fixed, key=lambda method: summary[method]["mean_J10"])
    eligible_fixed = [method for method in fixed if summary[method]["eligible"]]
    best_eligible = min(eligible_fixed, key=lambda method: summary[method]["mean_J10"]) if eligible_fixed else None
    summary["best_observed_fixed"] = best_observed
    summary["best_eligible_fixed"] = best_eligible
    summary["schedule_vs_best_observed_fixed_reduction"] = 1.-summary["schedule_beta"]["mean_J10"]/summary[best_observed]["mean_J10"]
    summary["schedule_vs_best_eligible_fixed_reduction"] = (
        1.-summary["schedule_beta"]["mean_J10"]/summary[best_eligible]["mean_J10"]
        if best_eligible else None)
    summary["RL_vs_best_observed_fixed_reduction"] = 1.-summary["RL_beta"]["mean_J10"]/summary[best_observed]["mean_J10"]
    summary["paired_task_results"] = [{"task_id": task["task_id"], **{
        method: next(row["J10"] for row in rows if row["task_id"] == task["task_id"] and row["method"] == method)
        for method in (best_observed, "schedule_beta", "RL_beta")}} for task in tasks]
    (OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# Dynamic β phase-switch screen", "", "Exploratory only; the schedule was chosen from the preceding frequency screen.",
             "", f"Best observed fixed value: `{best_observed}` (eligible: {best_eligible is not None}).", "",
             f"Scheduled β reduction vs best observed fixed: {summary['schedule_vs_best_observed_fixed_reduction']:.3%}.",
             f"Scheduled β reduction vs best eligible fixed: {summary['schedule_vs_best_eligible_fixed_reduction']}.",
             f"Existing RL reduction vs best observed fixed: {summary['RL_vs_best_observed_fixed_reduction']:.3%}.", "",
             "| Method | Complete | Eligible | QP failures | Saturations (command/actual) | Mean J10 | Low-stage J10 | High-stage J10 |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for method in methods:
        item = summary[method]
        lines.append(f"| {method} | {item['completed']}/{item['tasks']} | {item['eligible']} | {item['qp_failures']} | {item['command_saturation_cycles']}/{item['actual_saturation_steps']} | {item['mean_J10']:.8g} | {item['mean_J10_low_stages']:.8g} | {item['mean_J10_high_stage']:.8g} |")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
