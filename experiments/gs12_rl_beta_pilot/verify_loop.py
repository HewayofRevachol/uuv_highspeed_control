"""Check the incremental beta episode against frozen old traces and run_one."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.gs12_predictor_route.run_predictor import task_rows
from experiments.gs12_weight_route.runner_core import run_one
from experiments.gs12_joint_route.joint_reference import joint_task_reference
from .episode_beta import ACTION_BETAS, BetaEpisode
from .reward import external_j_track

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "experiments/gs12_predictor_route/output"
NUMERIC_FIELDS = ("command", "state_substep", "actual_substep",
                  "reference_substep", "beta_t")
TOLERANCE = 2e-7


def compare(actual, expected, controls):
    differences = {}
    for key in NUMERIC_FIELDS:
        n = controls if key in ("command", "beta_t") else controls*5
        if len(actual[key]) != n or len(expected[key]) < n:
            raise AssertionError(f"{key}: wrong length")
        difference = float(np.max(np.abs(actual[key]-expected[key][:n])))
        differences[key] = difference
        if not np.isfinite(difference) or difference > TOLERANCE:
            raise AssertionError(f"{key} differs by {difference}")
    if not np.array_equal(actual["qp_ok"], expected["qp_ok"][:controls]):
        raise AssertionError("QP validity differs")
    return differences


def archived_fixed_beta(archive):
    task = next(t for t in task_rows() if t["task_id"] == "ps_dev_depth_low")
    runs = [json.loads(p.read_text()) for p in (archive/"runs").glob("*.json")]
    checks = {}
    for beta in (0., .5, .75):
        matching = [r for r in runs if r.get("task_id") == task["task_id"]
                    and r.get("beta_or_policy") == beta]
        if len(matching) != 1:
            raise RuntimeError(f"need one archived {task['task_id']} beta={beta} run")
        trace = archive/"traces"/(matching[0]["run_id"]+".npz")
        episode = BetaEpisode(task, j_f75=1e-9, initial_beta=beta)
        obs = episode.observe()
        if obs.shape != (19,) or not np.isfinite(obs).all():
            raise AssertionError("invalid initial observation")
        _, _, done, info = episode.step(ACTION_BETAS.index(beta))
        if done or info["end_s"] != 8.:
            raise AssertionError("fixed beta first segment ended early")
        with np.load(trace) as old:
            checks[str(beta)] = compare(episode.result()["arrays"], old, 80)
            if beta == .75:
                with (archive/"dev_static_metrics.csv").open() as handle:
                    metrics = list(csv.DictReader(handle))
                recorded = next(row for row in metrics
                                if row["run_id"] == matching[0]["run_id"])
                score_delta = abs(external_j_track(old)-float(recorded["J_track"]))
                if score_delta > 1e-13:
                    raise AssertionError(f"external J_track differs by {score_delta}")
                checks[str(beta)]["archived_J_track_difference"] = score_delta
        checks[str(beta)]["archived_run_id"] = matching[0]["run_id"]
    return checks


def continuous_switch():
    task = next(t for t in task_rows() if t["task_id"] == "ps_dev_mixed")
    episode = BetaEpisode(task, j_f75=1e-9)
    episode.step(3)  # .75 for the first 8 s
    _, _, done, second = episode.step(2)  # slew toward .5 for the next 8 s
    if done or episode.session.k != 160:
        raise AssertionError("episode did not preserve a 16 s session")
    def policy(time_s, _grid, _task):
        k = round(time_s/.1)
        return .75 if k < 80 else max(.5, .75-.02*(k-79))
    old = run_one(task, "C4", 1.4, .7, 16., joint_task_reference,
                  (1., 1.), beta_policy=policy, preview_s=16.)
    current = episode.result()["arrays"]
    differences = compare(current, old["arrays"], 160)
    if np.max(np.abs(np.diff(current["beta_t"]))) > .02+1e-12:
        raise AssertionError("beta slew exceeded .02 per solve")
    if abs(current["beta_t"][-1]-.5) > 1e-12:
        raise AssertionError("beta did not reach .5")
    differences["second_segment_end_s"] = second["end_s"]
    return differences


def archived_static_path():
    archive = ROOT/"experiments/gs12_joint_route/output"
    tasks = [json.loads(line) for line in (archive/"tasks.jsonl").read_text().splitlines()]
    task = next(t for t in tasks if t["task_id"] == "dev_depth_b1")
    runs = [json.loads(p.read_text()) for p in (archive/"runs").glob("*.json")]
    matching = [r for r in runs if r.get("task_id") == task["task_id"]
                and r.get("controller_id") == "B"]
    if len(matching) != 1:
        raise RuntimeError("need one archived static C4/P0/W1 run")
    old = np.load(archive/"traces"/(matching[0]["run_id"]+".npz"))
    fresh = run_one(task, "C4", 1.4, .7, 8., joint_task_reference, (1., 1.))
    differences = {}
    for key in ("command", "state_substep", "actual_substep"):
        now = fresh["arrays"][key]
        diff = float(np.max(np.abs(now-old[key][:len(now)])))
        if diff > TOLERANCE:
            raise AssertionError(f"static {key} differs by {diff}")
        differences[key] = diff
    differences["archived_run_id"] = matching[0]["run_id"]
    return differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = {"archived_static_first_8s": archived_static_path(),
              "archived_fixed_beta_first_8s": archived_fixed_beta(args.archive),
              "continuous_75_to_50_16s": continuous_switch(),
              "tolerance": TOLERANCE}
    if args.out is not None:
        destination = args.out/"smoke"/"verify_loop.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
