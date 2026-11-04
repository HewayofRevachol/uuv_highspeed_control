"""Run the frozen six-case H15 dev comparison after Gate2 selection."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.h15_mainline.audit_gate2_training import (
    FROZEN_FILES, check_summary, read, sha)
from experiments.h15_mainline.train_masked_ppo import run_episode


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("B0_shared", "B2_gov", "B3_RL")
FIELDS = (
    "case", "plant_seed", "method", "completed", "strict_mainline",
    "abort_reason", "first_physical_violation", "periods",
    "backup_periods", "empty_candidate_periods", "final_progress_m",
    "max_abs_lateral_m", "max_abs_depth_m", "max_abs_speed_error_mps",
    "mean_tangent_water_speed_mps", "p50_full_period_ms",
    "p95_full_period_ms", "p99_full_period_ms", "max_full_period_ms",
    "source_directory", "source_summary_sha256")


def row_for(directory, method, case_id, plant_seed):
    checked = check_summary(directory, method, case_id, plant_seed)
    summary = read(directory / "summary.json")
    cycles = [json.loads(line) for line in
              (directory / "cycle_log.jsonl").read_text().splitlines()]
    timing = summary["timing_ms"]["full_period"]
    return {
        "case": case_id, "plant_seed": plant_seed, "method": method,
        **checked,
        "empty_candidate_periods": sum(row["candidate_empty"] for row in cycles),
        "mean_tangent_water_speed_mps": sum(
            row["actual_tangent_water_speed_mps"] for row in cycles) / len(cycles),
        "p50_full_period_ms": timing["p50"],
        "p95_full_period_ms": timing["p95"],
        "p99_full_period_ms": timing["p99"],
        "max_full_period_ms": timing["max"],
        "source_directory": str(directory),
        "source_summary_sha256": sha(directory / "summary.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate2-audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = read(args.gate2_audit)
    choice = read(args.selection)
    if audit["scope"] != "Gate2 frozen PPO train/dev" or choice["test_used"]:
        raise ValueError("Gate2 audit and dev-only selection required")
    selected = next(
        (row for row in choice["seed_results"]
         if row["seed"] == choice["selected_seed"]), None)
    if selected is None or sha(Path(choice["checkpoint"])) != choice["checkpoint_sha256"]:
        raise ValueError("selected checkpoint differs")
    if (selected["checkpoint_sha256"] != choice["checkpoint_sha256"] or
            not any(row["policy_seed"] == choice["selected_seed"] and
                    row["selected_checkpoint_sha256"] ==
                    choice["checkpoint_sha256"] for row in audit["seeds"])):
        raise ValueError("Gate2 audit and selected policy differ")
    freeze = read(FROZEN_FILES["freeze_sha256"])
    dev_path = ROOT / freeze["manifests"]["dev"]["path"]
    if sha(dev_path) != freeze["manifests"]["dev"]["sha256"]:
        raise ValueError("frozen dev manifest changed")
    manifest = read(dev_path)
    if len(manifest["cases"]) != 2 or len(manifest["seeds"]) != 3:
        raise ValueError("expected two dev geometries and three paired seeds")
    identity = {
        "schema": 1, "scope": "Gate3 paired dev quick screen",
        "gate2_audit_sha256": sha(args.gate2_audit),
        "selection_sha256": sha(args.selection),
        "dev_manifest_sha256": sha(dev_path),
        "selected_checkpoint_sha256": choice["checkpoint_sha256"],
        "selected_seed": choice["selected_seed"],
        "python_executable": "/bin/python3",
        "test_used": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    identity_path = args.out / "run_manifest.json"
    if identity_path.exists():
        if read(identity_path) != identity:
            raise ValueError("resume identity differs")
    elif any(args.out.iterdir()):
        raise ValueError("output directory is not empty")
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    episode_args = SimpleNamespace(
        freeze=FROZEN_FILES["freeze_sha256"],
        checkpoint=FROZEN_FILES["estimator_checkpoint_sha256"],
        config=FROZEN_FILES["candidate_config_sha256"],
        reward=FROZEN_FILES["reward_config_sha256"],
        pilot_periods=0)
    rows = []
    for case in manifest["cases"]:
        for plant_seed in manifest["seeds"]:
            for method in METHODS:
                directory = args.out / (
                    f"{case['id']}_s{plant_seed}_{method}")
                if not directory.exists():
                    run_episode(episode_args, "dev", case["id"],
                                plant_seed, method, directory,
                                Path(choice["checkpoint"]) if method == "B3_RL"
                                else None)
                if method == "B3_RL":
                    mode = read(directory / "policy_mode.json")
                    if mode["checkpoint_sha256"] != choice["checkpoint_sha256"]:
                        raise ValueError(f"wrong selected policy: {directory}")
                rows.append(row_for(directory, method, case["id"], plant_seed))
                print(f"{case['id']} {plant_seed} {method}: audited", flush=True)
    csv_path = args.out / "dev_paired_results.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


if __name__ == "__main__":
    main()
