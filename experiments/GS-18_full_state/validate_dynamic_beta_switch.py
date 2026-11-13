"""Evaluate the frozen beta schedule on phase-held-out switching paths."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from run_dynamic_beta_switch import CHECKPOINT, evaluate, load_policy


ROOT = Path(__file__).resolve().parents[2]
TASKS = Path(__file__).with_name("tasks_dynamic_beta_switch_validation_v1.jsonl")
SELECTION = Path(__file__).with_name("dynamic_beta_switch_selection_v1.json")
OUT = ROOT / "outputs/gs18_dynamic_beta_switch_validation_v1"
METHODS = ("beta_0.6", "schedule_beta", "RL_beta")


def input_hashes():
    paths = [TASKS, SELECTION, CHECKPOINT, CHECKPOINT.parent/"run.json",
             Path(__file__).resolve(), Path(__file__).with_name("run_dynamic_beta_switch.py"),
             Path(__file__).with_name("metrics.py"), Path(__file__).with_name("traced_env.py"),
             ROOT/"experiments/gs12_joint_route/joint_reference.py",
             ROOT/"experiments/gs12_rl_beta_gpu_admm/environment.py",
             ROOT/"experiments/gs12_rl_beta_gpu_admm/dynamics.py",
             ROOT/"experiments/gs12_rl_beta_gpu_admm/mpc.py"]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError(f"output directory must be empty: {OUT}")
    tasks = [json.loads(line) for line in TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    run = json.loads((CHECKPOINT.parent/"run.json").read_text(encoding="utf-8"))
    settings = {"predictor": run["predictor"], **run["admm"]}
    policy = load_policy(CHECKPOINT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"traces").mkdir()
    (OUT/"protocol.json").write_text(json.dumps({
        "selection": selection, "tasks": str(TASKS.resolve()),
        "checkpoint": str(CHECKPOINT.resolve()), "settings": settings,
        "methods": METHODS, "no_validation_reranking": True,
        "input_sha256": input_hashes()
    }, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (OUT/"tasks.jsonl").write_bytes(TASKS.read_bytes())
    rows = []
    for method in METHODS:
        rows.extend(evaluate(tasks, method, policy, settings, OUT/"traces"))
        print(f"{method}: {sum(row['method'] == method for row in rows)}/{len(tasks)}", flush=True)
    (OUT/"episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        completed = sum(row["status"].startswith("complete") for row in subset)
        summary[method] = {"completed": completed, "tasks": len(subset),
                           "qp_failures": sum(row["qp_failures"] for row in subset),
                           "command_saturation_cycles": sum(row["command_saturation_cycles"] for row in subset),
                           "actual_saturation_steps": sum(row["actual_saturation_steps"] for row in subset),
                           "eligible": completed == len(subset) and not any(
                               row["qp_failures"] or row["command_saturation_cycles"] or
                               row["actual_saturation_steps"] for row in subset),
                           "mean_J10": float(np.mean([row["J10"] for row in subset]))}
    base = summary["beta_0.6"]["mean_J10"]
    summary["schedule_reduction_vs_frozen_beta_0.6"] = 1.-summary["schedule_beta"]["mean_J10"]/base
    summary["RL_reduction_vs_frozen_beta_0.6"] = 1.-summary["RL_beta"]["mean_J10"]/base
    summary["schedule_wins"] = sum(
        next(row["J10"] for row in rows if row["method"] == "schedule_beta" and row["task_id"] == task["task_id"])
        < next(row["J10"] for row in rows if row["method"] == "beta_0.6" and row["task_id"] == task["task_id"])
        for task in tasks)
    summary["per_task"] = [{"task_id": task["task_id"], **{
        method: next(row["J10"] for row in rows if row["task_id"] == task["task_id"] and row["method"] == method)
        for method in METHODS}} for task in tasks]
    (OUT/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# Dynamic β phase-switch held-out evaluation", "",
             "The schedule and fixed baseline were frozen before these phase variants were run.", "",
             f"Dynamic schedule reduction vs β=0.6: {summary['schedule_reduction_vs_frozen_beta_0.6']:.3%}.",
             f"Paired wins: {summary['schedule_wins']}/{len(tasks)}.",
             f"RL reduction vs β=0.6: {summary['RL_reduction_vs_frozen_beta_0.6']:.3%}.", "",
             "| Method | Complete | Eligible | QP failures | Command/actual saturation | Mean J10 |",
             "| --- | ---: | --- | ---: | ---: | ---: |"]
    for method in METHODS:
        item = summary[method]
        lines.append(f"| {method} | {item['completed']}/{item['tasks']} | {item['eligible']} | {item['qp_failures']} | {item['command_saturation_cycles']}/{item['actual_saturation_steps']} | {item['mean_J10']:.8g} |")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
