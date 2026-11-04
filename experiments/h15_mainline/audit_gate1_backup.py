"""Audit the complete Gate1 smoke logs after enabling the shared backup."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = (("simple", 31001, "B0_shared", "b0"),
        ("coupled", 32001, "B0_shared", "b0"),
        ("coupled", 32001, "B2_gov", "b2"),
        ("mismatch", 32002, "B0_shared", "b0"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def close(left, right):
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0., abs_tol=1e-7)
        for a, b in zip(left, right))


def audit(case, seed, method, short, suite):
    directory = (ROOT / "outputs/experiments" /
                 (f"gov_uuv_h15_smoke_s{seed}" if suite == "gate1" else
                  f"e1gov_uuv_h15_smoke_s{seed}") /
                 (f"backup_full_{short}_{case}_20261104" if suite == "gate1" else
                  f"fullplan_{short}_{case}_20261104" if suite == "fullplan" else
                  f"full_{short}_{case}_20261104"))
    required = ("summary.json", "cycle_log.jsonl", "substep_samples.json",
                "backup_log.json", "backup_mode.json", "run.json", "console.log")
    if suite in ("e1", "fullplan"):
        required += ("estimator_mode.json", "estimator_log.jsonl")
    if any(not (directory / name).is_file() for name in required):
        raise ValueError(f"incomplete run: {directory}")
    summary = json.loads((directory / "summary.json").read_text())
    record = json.loads((directory / "run.json").read_text())
    cycles = [json.loads(line) for line in
              (directory / "cycle_log.jsonl").read_text().splitlines()]
    substeps = json.loads((directory / "substep_samples.json").read_text())
    backups = json.loads((directory / "backup_log.json").read_text())
    mode = json.loads((directory / "backup_mode.json").read_text())
    if (summary["case"], summary["seed"], summary["method"]) != (case, seed, method):
        raise ValueError(f"wrong run identity: {directory}")
    if mode["mode"] != "one_step_verified":
        raise ValueError(f"wrong backup mode: {directory}")
    sources = {
        "backup_mode.json": directory / "backup_mode.json",
        "estimator_mode.json": directory / "estimator_mode.json",
        "candidate_filter_config.json": ROOT / "experiments/h15_mainline" / (
            "candidate_filter_fullplan_config.json" if suite == "fullplan" else
            "candidate_filter_config.json"),
        "checkpoint.json": ROOT / "outputs/experiments/e1_uuv_h15_estimator_s31001" / (
            "freeze_fullplan_20261104/checkpoint.json" if suite == "fullplan" else
            "freeze_h35_20261104/checkpoint.json"),
        "quick_run_manifest.json": ROOT / "outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104/quick_run_manifest.json",
    }
    for name, expected in record["config_sha256"].items():
        source = sources.get(name, ROOT / "experiments/h15_mainline" / name)
        if sha(source) != expected:
            snapshots = (
                ROOT / "outputs/experiments/gov_uuv_h15_smoke_s31001/source_snapshots_20261104",
                ROOT / "outputs/experiments/e1gov_uuv_h15_smoke_s32001/source_snapshots_fullplan_20261104")
            if not any(sha(snapshot) == expected for directory in snapshots
                       for snapshot in directory.glob("*.py")):
                raise ValueError(f"changed source without snapshot: {source}")
    manifest = ROOT / "experiments/h15_mainline/manifests" / (
        "train.json" if case == "simple" else "dev.json")
    path = ROOT / "data/datasets/h15_mainline_quickstart" / {
        "simple": "train_turn_wide.csv", "coupled": "dev_coupled.csv",
        "mismatch": "dev_reversal.csv"}[case]
    for name, source in (("path.csv", path),
                         ("train_manifest.json" if case == "simple" else
                          "dev_manifest.json", manifest)):
        if sha(source) != record["data_sha256"][name]:
            raise ValueError(f"changed input: {source}")
    if (not summary["completed"] or summary["abort_reason"] is not None or
            len(cycles) != summary["periods"] or len(substeps) != 5*len(cycles) or
            not finite(cycles) or not finite(substeps) or not finite(backups)):
        raise ValueError(f"incomplete or nonfinite episode: {directory}")
    for index, row in enumerate(cycles):
        if row["period"] != index or row["fallback"] not in (
                None, "one_step_verified_fixed15"):
            raise ValueError(f"invalid cycle {index}: {directory}")
        if index:
            previous = cycles[index-1]
            if (not close(row["state_before"], previous["state_after"]) or
                    not close(row["actuator_before"], previous["actuator_after"]) or
                    not close(row["command_queue_before"][0], previous["issued_command"]) or
                    abs(row["ramped_nu_ref_mps"]-previous["ramped_nu_ref_mps"]) > .05000001):
                raise ValueError(f"timing or reference discontinuity at {index}: {directory}")
    backup_periods = [row["period"] for row in cycles
                      if row["fallback"] == "one_step_verified_fixed15"]
    if (len(backup_periods) != summary["backup_periods"] or
            [item["period"] for item in backups] != backup_periods or
            any(not item["accepted"] for item in backups) or
            bool(backup_periods) != bool(record["results"]["completed_with_backup"]) or
            summary["first_physical_violation"] is not None):
        raise ValueError(f"backup or physical verdict mismatch: {directory}")
    estimator_sources = None
    if suite in ("e1", "fullplan"):
        estimator_mode = json.loads((directory / "estimator_mode.json").read_text())
        estimator_rows = [json.loads(line) for line in
                          (directory / "estimator_log.jsonl").read_text().splitlines()]
        estimator_sources = {source: sum(row["source"] == source
                                         for row in estimator_rows)
                             for source in ("E0", "E1")}
        if (estimator_mode["mode"] != "E1" or len(estimator_rows) != len(cycles) or
                not finite(estimator_rows) or
                estimator_sources["E1"] == 0 or
                estimator_sources != summary["estimator_source_counts"] or
                any(row["period"] != index or row["samples"] > 10
                    for index, row in enumerate(estimator_rows))):
            raise ValueError(f"estimator history mismatch: {directory}")
        if suite == "fullplan":
            checkpoint = json.loads(sources["checkpoint.json"].read_text())
            parameters = checkpoint["parameters"]
            if estimator_mode["checkpoint_sha256"] != sha(sources["checkpoint.json"]):
                raise ValueError(f"wrong estimator checkpoint: {directory}")
            for index, row in enumerate(estimator_rows):
                if (not parameters["gain_bounds"][0] <= row["fin_gain"] <=
                        parameters["gain_bounds"][1] or
                        not parameters["tau_scale_bounds"][0] <= row["fin_tau_scale"] <=
                        parameters["tau_scale_bounds"][1] or
                        row["fallback_reason"] and
                        row["fallback_reason"].startswith("invalid_history")):
                    raise ValueError(f"invalid estimator value at {index}: {directory}")
                if index and row["source"] == estimator_rows[index-1]["source"] == "E1":
                    prior = estimator_rows[index-1]
                    if (abs(row["fin_gain"]-prior["fin_gain"]) >
                            parameters["max_gain_change_per_period"]+1e-10 or
                            abs(row["fin_tau_scale"]-prior["fin_tau_scale"]) >
                            parameters["max_tau_change_per_period"]+1e-10):
                        raise ValueError(f"unbounded estimator change at {index}: {directory}")
    return {"directory": str(directory), "method": method,
            "periods": len(cycles), "substeps": len(substeps),
            "backup_periods": len(backup_periods),
            "candidate_empty": summary["candidate_empty"],
            "strict_hist": summary["strict_hist"],
            "strict_mainline": summary["strict_mainline"],
            "final_progress_m": summary["final_progress_m"],
            "path_length_m": summary["path_length_m"],
            "max_abs_lateral_m": summary["max_abs_lateral_m"],
            "max_abs_depth_m": summary["max_abs_depth_m"],
            "max_abs_speed_error_mps": summary["max_abs_speed_error_mps"],
            "estimator_source_counts": estimator_sources,
            "selected_intents_mps": summary["selected_intents_mps"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=("gate1", "e1", "fullplan"),
                        default="gate1")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    runs = RUNS if args.suite == "gate1" else RUNS[1:3]
    result = {f"{case}_{short}": audit(case, seed, method, short, args.suite)
              for case, seed, method, short in runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {key: item[key] for key in (
        "periods", "backup_periods", "strict_mainline")}
                      for name, item in result.items()}, indent=2))


if __name__ == "__main__":
    main()
