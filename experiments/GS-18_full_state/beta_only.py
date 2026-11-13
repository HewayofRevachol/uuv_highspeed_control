"""Develop and validate beta alone, using the original MPC objective."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
BETAS = (0., .25, .5, .525, .55, .575, .6, .65, .75)
BASELINE = "beta_0.55"
CHECKPOINT = ROOT / "outputs/gs12_gpu_admm_20261112T051540Z/formal_1h/final_actor.pt"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tasks(path):
    tasks = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    ids = [task["task_id"] for task in tasks]
    if not tasks or len(ids) != len(set(ids)):
        raise ValueError("tasks must be nonempty and have unique task_id values")
    if any(task["duration_s"] != 64. for task in tasks):
        raise ValueError("the recorder and J10 require 64-second tasks")
    return tasks


def source_files():
    # Freeze the numerical implementation along with data and solver settings.
    paths = [DIRECTORY / name for name in ("beta_only.py", "run.py", "metrics.py", "traced_env.py")]
    paths += list((ROOT / "experiments/gs12_rl_beta_gpu_admm").glob("*.py"))
    paths += [ROOT / "experiments/gs12_joint_route/joint_reference.py",
              ROOT / "experiments/gs12_weight_route/reference_tasks.py"]
    return sorted(paths)


def protocol(args):
    tasks, validation = read_tasks(args.tasks), read_tasks(args.validation_tasks)
    # A renamed copy of a development/previous holdout task is not a fresh task.
    def signature(task):
        return json.dumps({key: value for key, value in task.items()
                           if key not in ("task_id", "split", "window_s")}, sort_keys=True)
    previous = tasks + read_tasks(DIRECTORY / "tasks_holdout.jsonl") + read_tasks(
        DIRECTORY / "tasks_validation_v2.jsonl")
    if set(map(signature, previous)) & set(map(signature, validation)):
        raise ValueError("validation tasks repeat a development or previously opened task")
    run_file = args.checkpoint.parent / "run.json"
    training = json.loads(run_file.read_text(encoding="utf-8"))
    frozen_files = [args.tasks.resolve(), args.validation_tasks.resolve(),
                    args.checkpoint.resolve(), run_file.resolve(), *source_files()]
    return {"experiment": "GS-18 beta only v1", "betas": BETAS,
            "tasks": str(args.tasks.resolve()),
            "validation_tasks": str(args.validation_tasks.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "settings": {"predictor": training["predictor"], **training["admm"]},
            "controls": {"fixed_initial_beta": .75, "attitude_scale": 1.,
                         "velocity_scale": 1., "position_scale": 1.},
            "baseline_method": BASELINE, "min_improvement_fraction": .05,
            "acceptance": "all tasks complete, zero QP failures and saturation; mean J10 reduction >= 5%",
            "score": "unchanged metrics.py J10, 0.02-64 s; task-equal mean",
            "batch_size": args.batch_size,
            "files_sha256": {str(path): digest(path) for path in frozen_files}}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def method_stats(rows, methods, task_ids):
    expected = {(method, task_id) for method in methods for task_id in task_ids}
    actual = [(row["method"], row["task_id"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("results must contain exactly one row per method and task")
    result = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        complete = [row for row in subset if row["status"] == "complete_valid"
                    and row["J10"] is not None and math.isfinite(row["J10"])]
        events = {key: sum(row[key] for row in subset) for key in
                  ("qp_failures", "command_saturation_cycles", "actual_saturation_steps")}
        result[method] = {"tasks": len(subset), "complete": len(complete), **events,
                          "eligible": len(complete) == len(subset) and not any(events.values()),
                          "mean_J10_complete_only": fmean(row["J10"] for row in complete) if complete else None,
                          "mean_rmse_complete_only": {
                              key: fmean(row["rmse"][key] for row in complete)
                              for key in complete[0]["rmse"]} if complete else None,
                          "mean_groups_complete_only": {
                              key: fmean(row["groups"][key] for row in complete)
                              for key in complete[0]["groups"]} if complete else None}
    return result


def compare(rows, stats, baseline, candidate):
    if not stats[baseline]["eligible"] or not stats[candidate]["eligible"]:
        return None
    index = {(row["method"], row["task_id"]): row for row in rows}
    pairs = [(index[baseline, row["task_id"]], row) for row in rows if row["method"] == candidate]
    base = stats[baseline]["mean_J10_complete_only"]
    value = stats[candidate]["mean_J10_complete_only"]
    return {"baseline": baseline, "candidate": candidate,
            "reduction_fraction": 1. - value / base if base > 0 else None,
            "wins": sum(right["J10"] < left["J10"] for left, right in pairs),
            "rmse_change": {key: stats[candidate]["mean_rmse_complete_only"][key] - value
                            for key, value in stats[baseline]["mean_rmse_complete_only"].items()}}


def analyze_development(rows, methods, task_ids):
    stats = method_stats(rows, methods, task_ids)
    eligible = [method for method in methods if method.startswith("beta_") and stats[method]["eligible"]]
    best = min(eligible, key=lambda method: stats[method]["mean_J10_complete_only"]) if eligible else None
    oracle = None
    if best:
        index = {(row["method"], row["task_id"]): row["J10"] for row in rows}
        per_task = {task: min(eligible, key=lambda method: index[method, task]) for task in task_ids}
        oracle_mean = fmean(index[method, task] for task, method in per_task.items())
        best_mean = stats[best]["mean_J10_complete_only"]
        oracle = {"mean_J10": oracle_mean, "per_task_best_fixed": per_task,
                  "reduction_fraction": 1. - oracle_mean / best_mean if best_mean > 0 else 0.,
                  "interpretation": "taskwise constant-beta diagnostic, NOT a bound on time-varying beta"}
    return {"methods": stats, "best_fixed": best, "taskwise_fixed_oracle": oracle,
            "best_fixed_vs_beta055": compare(rows, stats, BASELINE, best) if best else None,
            "RL_vs_best_fixed": compare(rows, stats, best, "RL_beta") if best else None}


def analyze_validation(rows, methods, task_ids, candidate, min_improvement):
    stats = method_stats(rows, methods, task_ids)
    comparison = compare(rows, stats, BASELINE, candidate)
    improvement = comparison["reduction_fraction"] if comparison else None
    return {"methods": stats, "candidate_method": candidate,
            "candidate_vs_baseline": comparison,
            "RL_vs_frozen_fixed": compare(rows, stats, candidate, "RL_beta"),
            "accepted": improvement is not None and improvement >= min_improvement}


def evaluate_methods(methods, tasks, frozen, output):
    # Import GPU code only when an actual experiment is requested, after preflight.
    import torch
    from run import evaluate, load_policy

    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for evaluation")
    torch.set_num_threads(1)
    policy = load_policy(Path(frozen["checkpoint"]))
    traces = output / "traces"
    traces.mkdir()
    rows = []
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for method in methods:
            batch = evaluate(method, tasks, policy, frozen["settings"], frozen["batch_size"],
                             traces, **frozen["controls"])
            rows.extend(batch)
            for row in batch:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    return rows


def write_report(output, summary):
    lines = ["# GS-18 β-only results", "", "MPC weight multipliers = 1; all methods start at beta=0.75.", "",
             "| Method | Complete | Eligible | Mean J10 (complete only) |",
             "| --- | ---: | --- | ---: |"]
    for method, stats in summary["methods"].items():
        value = stats["mean_J10_complete_only"]
        lines.append(f"| {method} | {stats['complete']}/{stats['tasks']} | {stats['eligible']} | {value} |")
    if "best_fixed" in summary:
        lines += ["", f"Best fixed beta: {summary['best_fixed']}", "",
                  "Best fixed beta vs beta=0.55:", "```json",
                  json.dumps(summary["best_fixed_vs_beta055"], indent=2), "```", "",
                  "Taskwise constant-beta diagnostic (not a bound on dynamic beta):",
                  "```json", json.dumps(summary["taskwise_fixed_oracle"], indent=2), "```"]
    else:
        lines += ["", "Frozen candidate vs beta=0.55:", "```json",
                  json.dumps(summary["candidate_vs_baseline"], indent=2), "```", "",
                  f"Passes frozen acceptance: {summary['accepted']}"]
    lines += ["", "See summary.json for all state/group means and comparisons; episodes.jsonl for each task's",
              "J10, groups, ten-state RMSE/peaks and quality events; traces/ for sampled trajectories.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    develop = sub.add_parser("develop", help="paired grid, current RL diagnostic, and fixed-beta selection")
    develop.add_argument("--tasks", type=Path, default=DIRECTORY / "tasks_dev.jsonl")
    develop.add_argument("--validation-tasks", type=Path, default=DIRECTORY / "tasks_beta_validation_v3.jsonl")
    develop.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    develop.add_argument("--batch-size", type=int, default=12)
    validate = sub.add_parser("validate", help="evaluate frozen fixed beta; never rerank on validation")
    validate.add_argument("--selection", type=Path, required=True)
    for command in (develop, validate):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--dry-run", action="store_true", help="check files and print protocol without simulation or output writes")
    args = parser.parse_args()
    try:
        if args.stage == "develop":
            if args.batch_size < 1:
                raise ValueError("batch size must be positive")
            frozen = protocol(args)
            methods = [f"beta_{beta:g}" for beta in BETAS] + ["RL_beta"]
            task_file = args.tasks
        else:
            selected = json.loads(args.selection.read_text(encoding="utf-8"))
            frozen = selected["protocol"]
            for filename, expected in frozen["files_sha256"].items():
                if digest(filename) != expected:
                    raise ValueError(f"frozen input changed: {filename}")
            methods = list(dict.fromkeys((frozen["baseline_method"], selected["candidate_method"], "RL_beta")))
            task_file = Path(frozen["validation_tasks"])
        tasks = read_tasks(task_file)
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError("output directory must be empty")
        if args.dry_run:
            print(json.dumps({"stage": args.stage, "methods": methods, "tasks": len(tasks),
                              "protocol": frozen}, ensure_ascii=False, indent=2))
            return
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "protocol.json", frozen)
        (args.out / "tasks.jsonl").write_bytes(task_file.read_bytes())
        rows = evaluate_methods(methods, tasks, frozen, args.out)
        task_ids = [task["task_id"] for task in tasks]
        if args.stage == "develop":
            summary = analyze_development(rows, methods, task_ids)
            if summary["best_fixed"]:
                write_json(args.out / "selected.json", {"candidate_method": summary["best_fixed"],
                                                        "protocol": frozen})
        else:
            candidate = selected["candidate_method"]
            summary = analyze_validation(rows, methods, task_ids, candidate,
                                         frozen["min_improvement_fraction"])
            write_json(args.out / "selection.json", selected)
        write_json(args.out / "summary.json", summary)
        write_report(args.out, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
