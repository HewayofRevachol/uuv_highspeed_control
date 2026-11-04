"""Audit the frozen three-seed H15 PPO train/dev protocol."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.h15_mainline.audit_rl_run import audit as audit_b3
from experiments.h15_mainline.run_record import validate as validate_run
from experiments.h15_mainline.train_masked_ppo import (
    POLICY_SEEDS, TRAIN_SCHEDULE, dev_score)


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("B0_shared", "B2_gov", "B3_RL")
FROZEN_FILES = {
    "freeze_sha256": ROOT / "outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json",
    "estimator_checkpoint_sha256": ROOT / "outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_fullplan_20261104/checkpoint.json",
    "candidate_config_sha256": ROOT / "experiments/h15_mainline/candidate_filter_fullplan_config.json",
    "reward_config_sha256": ROOT / "experiments/h15_mainline/reward_v1.json",
}


def reject_nonfinite(token):
    raise ValueError(f"nonfinite JSON value: {token}")


def read(path):
    return json.loads(path.read_text(), parse_constant=reject_nonfinite)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left, right):
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0., abs_tol=1e-7)
        for a, b in zip(left, right))


def check_summary(directory, method, case_id, plant_seed):
    summary = read(directory / "summary.json")
    record = validate_run(read(directory / "run.json"))
    if (summary["method"], summary["case"], summary["seed"]) != (
            method, case_id, plant_seed):
        raise ValueError(f"wrong episode identity: {directory}")
    if record["method"] != method or record["seeds"] != [plant_seed]:
        raise ValueError(f"wrong run record identity: {directory}")
    for filename, key in (
            ("candidate_filter_config.json", "candidate_config_sha256"),
            ("checkpoint.json", "estimator_checkpoint_sha256"),
            ("quick_run_manifest.json", "freeze_sha256")):
        if record["config_sha256"][filename] != sha(FROZEN_FILES[key]):
            raise ValueError(f"run used different frozen input: {directory}")
    if summary["limited_diagnostic"]:
        raise ValueError(f"truncated diagnostic episode: {directory}")
    cycles = [json.loads(line, parse_constant=reject_nonfinite) for line in
              (directory / "cycle_log.jsonl").read_text().splitlines()]
    substeps = read(directory / "substep_samples.json")
    if len(cycles) != summary["periods"] or len(substeps) != 5 * len(cycles):
        raise ValueError(f"incomplete episode record: {directory}")
    backups = 0
    for index, cycle in enumerate(cycles):
        fallback = cycle["fallback"]
        selected = cycle["selected_candidate_mps"]
        if cycle["period"] != index or fallback not in (
                None, "one_step_verified_fixed15") or (
                (selected is None) != (fallback is not None)):
            raise ValueError(f"selection or fallback missing at {directory}:{index}")
        if index:
            previous = cycles[index - 1]
            if (not close(cycle["state_before"], previous["state_after"]) or
                    not close(cycle["actuator_before"],
                              previous["actuator_after"]) or
                    not close(cycle["command_queue_before"][0],
                              previous["issued_command"]) or
                    abs(cycle["ramped_nu_ref_mps"] -
                        previous["ramped_nu_ref_mps"]) > .05000001):
                raise ValueError(f"state, queue or ramp discontinuity: {directory}:{index}")
        backups += fallback is not None
    if (backups != summary["backup_periods"] or
            bool(any(row["candidate_empty"] for row in cycles)) !=
            bool(summary["candidate_empty"])):
        raise ValueError(f"backup or empty-candidate count differs: {directory}")
    if any(substeps[i]["time_s"] <= substeps[i - 1]["time_s"]
           for i in range(1, len(substeps))):
        raise ValueError(f"substep time is not increasing: {directory}")
    if method == "B3_RL":
        mode = read(directory / "policy_mode.json")
        if mode["reward_config_sha256"] != sha(FROZEN_FILES["reward_config_sha256"]):
            raise ValueError(f"B3 used a different reward version: {directory}")
        checked = audit_b3(directory)
        if checked["periods"] != summary["periods"]:
            raise ValueError(f"B3 audit disagrees: {directory}")
    return {
        key: summary[key] for key in (
            "completed", "strict_mainline", "abort_reason",
            "first_physical_violation", "periods", "backup_periods",
            "final_progress_m", "max_abs_lateral_m", "max_abs_depth_m",
            "max_abs_speed_error_mps")
    }


def audit_seed(directory):
    run = read(directory / "run.json")
    seed = run["policy_seed"]
    if seed not in POLICY_SEEDS or run["scope"] != "frozen train/dev":
        raise ValueError(f"not a frozen policy seed: {directory}")
    if run["python_executable"] != "/bin/python3":
        raise ValueError(f"wrong interpreter: {directory}")
    for key, path in FROZEN_FILES.items():
        if run[key] != sha(path):
            raise ValueError(f"frozen input differs: {path}")
    if tuple(map(tuple, run["train_schedule"])) != TRAIN_SCHEDULE:
        raise ValueError(f"changed training schedule: {directory}")
    for name, expected in run["source_sha256"].items():
        if sha(ROOT / "experiments/h15_mainline" / name) != expected:
            raise ValueError(f"source changed during run: {name}")
    history = read(directory / "train_history.json")
    if len(history) != len(TRAIN_SCHEDULE):
        raise ValueError(f"incomplete train history: {directory}")
    train = []
    for index, (case_id, plant_seed) in enumerate(TRAIN_SCHEDULE):
        row = history[index]
        if (row["episode"], row["case_id"], row["plant_seed"]) != (
                index, case_id, plant_seed):
            raise ValueError(f"train order differs: {directory}")
        episodes = {}
        for method in METHODS:
            path = Path(row["runs"][method])
            episodes[method] = check_summary(path, method, case_id, plant_seed)
            if method == "B3_RL":
                expected = directory / (
                    "policy_initial.pt" if index == 0 else
                    f"policy_after_{index-1}.pt")
                if read(path / "policy_mode.json")["checkpoint_sha256"] != sha(expected):
                    raise ValueError(f"wrong training policy checkpoint: {path}")
            for key, value in row["results"][method].items():
                if episodes[method][key] != value:
                    raise ValueError(f"history differs from episode: {path}")
        checkpoint = directory / f"policy_after_{index}.pt"
        if sha(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"train checkpoint differs: {checkpoint}")
        train.append({"case": case_id, "seed": plant_seed,
                      "methods": episodes, "ppo": row["ppo"]})

    scores = read(directory / "dev_scores.json")
    selection = read(directory / "selection.json")
    if selection["dev_scores"] != scores:
        raise ValueError(f"selection scores differ: {directory}")
    dev_manifest = read(ROOT / "experiments/h15_mainline/manifests/dev.json")
    dev = {}
    for name, checkpoint in (
            ("initial", directory / "policy_initial.pt"),
            ("trained", directory / f"policy_after_{len(TRAIN_SCHEDULE)-1}.pt")):
        if scores[name]["checkpoint_sha256"] != sha(checkpoint):
            raise ValueError(f"dev checkpoint differs: {checkpoint}")
        episodes = []
        for case in dev_manifest["cases"]:
            for plant_seed in dev_manifest["seeds"]:
                path = directory / (
                    f"dev_{name}_{case['id']}_s{plant_seed}_B3_RL")
                episodes.append(check_summary(
                    path, "B3_RL", case["id"], plant_seed))
                if read(path / "policy_mode.json")["checkpoint_sha256"] != sha(checkpoint):
                    raise ValueError(f"wrong dev policy checkpoint: {path}")
        if scores[name]["episodes"] != len(episodes) or len(episodes) != 6:
            raise ValueError(f"incomplete dev evaluation: {directory}")
        if scores[name]["score"] != list(dev_score(episodes)):
            raise ValueError(f"dev score differs from episode summaries: {directory}")
        dev[name] = episodes
    choice = max(("initial", "trained"),
                 key=lambda name: tuple(scores[name]["score"]))
    selected = directory / (
        "policy_initial.pt" if choice == "initial" else
        f"policy_after_{len(TRAIN_SCHEDULE)-1}.pt")
    if (selection["selected"] != choice or
            Path(selection["checkpoint"]) != selected or
            selection["checkpoint_sha256"] != sha(selected)):
        raise ValueError(f"checkpoint selection differs: {directory}")
    return {
        "policy_seed": seed,
        "input_sha256": {name: run[name] for name in (
            "freeze_sha256", "estimator_checkpoint_sha256",
            "candidate_config_sha256", "reward_config_sha256")},
        "train": train, "dev": dev,
        "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": sha(selected),
        "dev_scores": scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = [audit_seed(directory) for directory in args.runs]
    if sorted(row["policy_seed"] for row in rows) != list(POLICY_SEEDS):
        raise ValueError("expected exactly the three frozen seeds")
    if any(row["input_sha256"] != rows[0]["input_sha256"] for row in rows[1:]):
        raise ValueError("three seeds used different frozen inputs")
    result = {"schema": 1, "scope": "Gate2 frozen PPO train/dev",
              "test_used": False, "seeds": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
