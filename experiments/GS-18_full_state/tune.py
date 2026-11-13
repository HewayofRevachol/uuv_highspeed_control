"""Continue GS-18 development tuning after removing the per-state RMSE gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run import evaluate, load_policy, read_jsonl, sha256


def run_config(config, tasks, policy, settings, output):
    output.mkdir(parents=True)
    traces = output/"traces"
    traces.mkdir()
    method = f"beta_{config['beta']:g}"
    rows = evaluate(method, tasks, policy, settings, len(tasks), traces,
                    attitude_scale=config["attitude_scale"],
                    position_scale=config["position_scale"])
    with (output/"episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row)+"\n")
    complete = [row for row in rows if row["J10"] is not None]
    result = {"config": config, "tasks": len(rows), "complete": len(complete),
              "qp_failures": sum(row["qp_failures"] for row in rows),
              "command_saturation_cycles": sum(row["command_saturation_cycles"] for row in rows),
              "actual_saturation_steps": sum(row["actual_saturation_steps"] for row in rows),
              "mean_J10": float(np.mean([row["J10"] for row in complete])) if complete else None,
              "mean_groups": {key: float(np.mean([row["groups"][key] for row in complete]))
                              for key in complete[0]["groups"]} if complete else None,
              "mean_rmse": {key: float(np.mean([row["rmse"][key] for row in complete]))
                            for key in complete[0]["rmse"]} if complete else None}
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def best_complete(results):
    return min((row for row in results if row["complete"] == row["tasks"] and
                row["qp_failures"] == 0 and row["command_saturation_cycles"] == 0 and
                row["actual_saturation_steps"] == 0), key=lambda row: row["mean_J10"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/gs12_gpu_admm_20261112T051540Z/formal_1h/final_actor.pt"))
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    directory = Path(__file__).parent
    task_file = directory/"tasks_dev.jsonl"
    tasks = read_jsonl(task_file)
    training_file = args.checkpoint.parent/"run.json"
    training = json.loads(training_file.read_text(encoding="utf-8"))
    settings = {"predictor": training["predictor"], **training["admm"]}
    provenance = {"tasks_sha256": sha256(task_file), "checkpoint_sha256": sha256(args.checkpoint),
                  "training_run_sha256": sha256(training_file), "settings": settings,
                  "selection": "task-equal mean J10, all tasks complete, no QP failures or saturation",
                  "removed_gate": "per-state mean RMSE regression tolerance",
                  "plan": "attitude 8,16,32,64; extend to 128 if still improving; beta 0,.25,.5,.75; position .25,1,4"}
    (args.out/"protocol.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    torch.set_num_threads(1)
    policy = load_policy(args.checkpoint)
    results = []

    def trial(beta, attitude, position=1.):
        config = {"beta": beta, "attitude_scale": attitude, "position_scale": position}
        name = f"beta{beta:g}_att{attitude:g}_pos{position:g}"
        result = run_config(config, tasks, policy, settings, args.out/name)
        results.append(result)
        (args.out/"development.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
        return result

    # Start at the previous candidate to keep all comparisons on this same runner.
    for attitude in (8., 16., 32., 64.):
        trial(.5, attitude)
    if best_complete(results)["config"]["attitude_scale"] == 64.:
        trial(.5, 128.)
    winner = best_complete(results)["config"].copy()
    for beta in (0., .25, .75):
        trial(beta, winner["attitude_scale"])
    winner = best_complete(results)["config"].copy()
    for position in (.25, 4.):
        trial(winner["beta"], winner["attitude_scale"], position)
    winner = best_complete(results)
    (args.out/"selected.json").write_text(json.dumps(winner, indent=2)+"\n", encoding="utf-8")
    # Freeze before evaluating either verification set; neither affects selection.
    for label, manifest in (("retest", "tasks_holdout.jsonl"),
                            ("fresh", "tasks_validation_v2.jsonl")):
        verify_file = directory/manifest
        verify_tasks = read_jsonl(verify_file)
        (args.out/f"{label}_tasks.jsonl").write_bytes(verify_file.read_bytes())
        for name, config in (("baseline", {"beta": .55, "attitude_scale": 1., "position_scale": 1.}),
                             ("candidate", winner["config"])):
            run_config(config, verify_tasks, policy, settings, args.out/f"{label}_{name}")


if __name__ == "__main__":
    main()
