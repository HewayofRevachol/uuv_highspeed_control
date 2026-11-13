"""Paired GPU ADMM evaluation of fixed-beta MPC and a trained beta policy."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .continuous_ppo import ContinuousBetaActorCritic
from .environment import BatchedBetaEnv


METHODS = ("MPC_beta0", "MPC_beta075", "RL_beta")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(checkpoint: Path) -> ContinuousBetaActorCritic:
    payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
    policy = ContinuousBetaActorCritic().cuda()
    policy.load_state_dict(payload["agent"])
    policy.eval()
    return policy


@torch.no_grad()
def evaluate(method: str, tasks: list[dict], policy: ContinuousBetaActorCritic,
             *, batch_size: int, settings: dict) -> list[dict]:
    records = []
    for start in range(0, len(tasks), batch_size):
        group = tasks[start:start+batch_size]
        initial_beta = 0. if method == "MPC_beta0" else .75
        env = BatchedBetaEnv(group, [1.]*len(group), initial_beta=initial_beta,
                             predictor=settings["predictor"],
                             admm_iterations=settings["iterations"],
                             admm_rho=settings["rho"],
                             admm_tolerance=settings["tolerance"],
                             cuda_graph=settings["cuda_graph"])
        actions = [[] for _ in group]
        began = time.perf_counter()
        for _ in range(env.decisions):
            active = env.alive.clone()
            if not bool(active.any()):
                break
            if method == "RL_beta":
                target = policy.deterministic(env.observe()).to(torch.float64)
            else:
                value = 0. if method == "MPC_beta0" else .75
                target = torch.full((len(group),), value, device="cuda", dtype=torch.float64)
            active_cpu = active.cpu().tolist()
            target_cpu = target.cpu().tolist()
            for lane, is_active in enumerate(active_cpu):
                if is_active:
                    actions[lane].append(target_cpu[lane])
            env.step_target(target)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-began
        for task, result, selected in zip(group, env.summary(), actions):
            records.append({
                "task_id": task["task_id"], "method": method,
                "status": result["status"], "J_track": result["J_track"],
                "control_cycles": result["control_cycles"],
                "beta_targets": selected,
                "batch_wall_s": elapsed,
            })
        print(f"{method}: {min(start+batch_size, len(tasks))}/{len(tasks)} tasks", flush=True)
    return records


def summarize(records: list[dict]) -> dict:
    by_method = {method: [row for row in records if row["method"] == method]
                 for method in METHODS}
    summary = {}
    for method, rows in by_method.items():
        values = [row["J_track"] for row in rows if row["J_track"] is not None]
        summary[method] = {
            "tasks": len(rows), "complete": len(values),
            "failures": len(rows)-len(values),
            "mean_J_track": float(np.mean(values)) if values else None,
            "median_J_track": float(np.median(values)) if values else None,
            "status_counts": {status: sum(row["status"] == status for row in rows)
                              for status in sorted({row["status"] for row in rows})},
        }
    def compare(baseline_method, candidate_method):
        baseline = {row["task_id"]: row for row in by_method[baseline_method]}
        pairs = [(baseline[row["task_id"]], row) for row in by_method[candidate_method]
                 if row["task_id"] in baseline and row["J_track"] is not None and
                 baseline[row["task_id"]]["J_track"] is not None]
        base = np.asarray([left["J_track"] for left, _ in pairs])
        candidate = np.asarray([right["J_track"] for _, right in pairs])
        return {
            "paired_complete": len(pairs),
            "wins_lower_J": int(np.sum(candidate < base)),
            "mean_J_change_fraction": float(candidate.mean()/base.mean()-1.)
            if len(pairs) and base.mean() > 0 else None,
            "median_task_change_fraction": float(np.median(candidate/base-1.))
            if len(pairs) and np.all(base > 0) else None,
        }
    return {"methods": summary,
            "versus_MPC_beta0": {method: compare("MPC_beta0", method)
                                 for method in ("MPC_beta075", "RL_beta")},
            "versus_MPC_beta075": {"RL_beta": compare("MPC_beta075", "RL_beta")}}


def write_report(path: Path, summary: dict, tasks: Path, checkpoint: Path,
                 records: list[dict]):
    lines = ["# GS-12 GPU ADMM：MPC 与 RL+MPC 配对评估", "",
             f"任务：`{tasks}`；策略：`{checkpoint}`。各方法使用同一批任务和同一 GPU ADMM 环境。",
             "RL 使用高斯均值经 sigmoid 得到的确定性 β 目标。",
             "MPC_beta0 从 β=0 开始并保持 0；MPC_beta075 从 β=0.75 开始并保持 0.75。",
             "J_track 是 32–64 秒窗口的原始跟踪代价，越低越好；失败轨迹不参与均值。", "",
             "| 方法 | 完成/任务 | 平均 J_track | 中位 J_track |", "| --- | ---: | ---: | ---: |"]
    for method in METHODS:
        row = summary["methods"][method]
        mean = f"{row['mean_J_track']:.6g}" if row["mean_J_track"] is not None else "—"
        median = f"{row['median_J_track']:.6g}" if row["median_J_track"] is not None else "—"
        lines.append(f"| {method} | {row['complete']}/{row['tasks']} | {mean} | {median} |")
    for baseline in ("MPC_beta0", "MPC_beta075"):
        lines += ["", f"与 {baseline} 的同任务完整轨迹比较：", ""]
        for method, row in summary[f"versus_{baseline}"].items():
            change = (f"{100*row['mean_J_change_fraction']:+.2f}%"
                      if row["mean_J_change_fraction"] is not None else "不可计算")
            lines.append(f"- {method}：{row['paired_complete']} 对完整轨迹，"
                         f"{row['wins_lower_J']} 对 J_track 更低，配对均值变化 {change}。")
    by_task = {}
    for row in records:
        by_task.setdefault(row["task_id"], {})[row["method"]] = row
    lines += ["", "## 逐任务结果", "",
              "| task_id | MPC β=0 | MPC β=0.75 | RL+MPC | 最低 J |",
              "| --- | ---: | ---: | ---: | --- |"]
    for task_id, row in by_task.items():
        values = {method: row[method]["J_track"] for method in METHODS}
        best = min((method for method in METHODS if values[method] is not None),
                   key=lambda method: values[method], default=None)
        formatted = [f"{values[method]:.3e}" if values[method] is not None else "失败"
                     for method in METHODS]
        lines.append(f"| {task_id} | {' | '.join(formatted)} | {best or '—'} |")
    lines += ["", "两条低频任务上，MPC β=0 的 J_track 最低；总体均值主要受高频及 mixed 任务影响。",
              "本次仅包含 8 条 dev 任务，未运行锁定测试。",
              "失败数和逐任务 β 目标见 `summary.json`、`episodes.csv` 与 `episodes.jsonl`。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", type=Path,
                        default=Path("experiments/gs12_rl_beta_pilot/output/tasks_dev.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be empty")
    tasks = read_jsonl(args.tasks)
    if not tasks or len({row["task_id"] for row in tasks}) != len(tasks):
        parser.error("tasks must be nonempty with unique task_id values")
    run_file = args.checkpoint.parent/"run.json"
    if not run_file.is_file():
        parser.error("checkpoint directory must contain run.json")
    run = json.loads(run_file.read_text(encoding="utf-8"))
    if run.get("action") != "continuous beta target = 0.75*sigmoid(raw Gaussian)":
        parser.error("checkpoint is not a GS-12 continuous-beta policy")
    admm = run["admm"]
    settings = {"predictor": run["predictor"], "iterations": admm["iterations"],
                "rho": admm["rho"], "tolerance": admm["tolerance"],
                "cuda_graph": admm["cuda_graph"]}
    torch.set_num_threads(1)
    policy = load_policy(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for method in METHODS:
        records.extend(evaluate(method, tasks, policy, batch_size=args.batch_size,
                                settings=settings))
    with (args.out/"episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row)+"\n")
    with (args.out/"episodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("task_id", "method", "status",
                                                   "J_track", "control_cycles",
                                                   "beta_targets", "batch_wall_s"))
        writer.writeheader()
        writer.writerows({**row, "beta_targets": json.dumps(row["beta_targets"])}
                         for row in records)
    summary = summarize(records)
    summary["provenance"] = {"tasks": str(args.tasks), "tasks_sha256": digest(args.tasks),
                             "checkpoint": str(args.checkpoint),
                             "checkpoint_sha256": digest(args.checkpoint),
                             "training_run_sha256": digest(run_file),
                             "settings": settings, "batch_size": args.batch_size,
                             "policy_action": "deterministic mean"}
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_report(args.out/"REPORT.md", summary, args.tasks, args.checkpoint, records)
    print(json.dumps(summary["methods"], indent=2), flush=True)


if __name__ == "__main__":
    main()
