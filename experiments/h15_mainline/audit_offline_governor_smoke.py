"""Audit the three paired Gate1 offline mechanism runs without changing inputs."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    ("simple", 31001, "full_direct"),
    ("coupled", 32001, "full"),
    ("mismatch", 32002, "full"),
)
METHODS = (("b0", "B0_shared"), ("b2", "B2_gov"))
ARTIFACTS = ("summary.json", "cycle_log.jsonl", "substep_samples.json",
             "run.json", "console.log")


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def close(a, b):
    return all(math.isclose(x, y, rel_tol=0., abs_tol=1e-7)
               for x, y in zip(a, b))


def audit_run(directory, method, case):
    missing = [name for name in ARTIFACTS if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"{directory}: missing {missing}")
    summary = json.loads((directory / "summary.json").read_text())
    record = json.loads((directory / "run.json").read_text())
    cycles = [json.loads(line) for line in
              (directory / "cycle_log.jsonl").read_text().splitlines()]
    substeps = json.loads((directory / "substep_samples.json").read_text())
    if (summary["method"], summary["case"]) != (method, case):
        raise ValueError(f"{directory}: wrong method or case")
    if not cycles or len(cycles) != summary["periods"]:
        raise ValueError(f"{directory}: cycle count mismatch")
    for name, expected in record["config_sha256"].items():
        source = ROOT / "experiments/h15_mainline" / name
        if not source.exists():
            source = ROOT / "outputs/experiments/mpc_uuv_h15_quickstart_s31001/freeze_20261104" / name
        if not source.exists():
            source = ROOT / "outputs/experiments/e1_uuv_h15_estimator_s31001/freeze_h35_20261104" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            snapshots = (ROOT / "outputs/experiments/gov_uuv_h15_smoke_s31001/"
                         "source_snapshots_20261104")
            if not any(hashlib.sha256(item.read_bytes()).hexdigest() == expected
                       for item in snapshots.glob("*.py")):
                raise ValueError(f"{directory}: source changed: {name}")
    manifest = ROOT / "experiments/h15_mainline/manifests" / (
        "train.json" if case == "simple" else "dev.json")
    path = ROOT / ("data/datasets/h15_mainline_quickstart/" + {
        "simple": "train_turn_wide.csv", "coupled": "dev_coupled.csv",
        "mismatch": "dev_reversal.csv"}[case])
    for name, source in (("path.csv", path),
                         ("train_manifest.json" if case == "simple" else
                          "dev_manifest.json", manifest)):
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["data_sha256"][name]:
            raise ValueError(f"{directory}: data changed: {name}")
    executed = 0
    for index, row in enumerate(cycles):
        if row["period"] != index or not finite(row):
            raise ValueError(f"{directory}: nonfinite or discontinuous cycle {index}")
        if index:
            prior = cycles[index - 1]
            if not close(row["state_before"], prior["state_after"]):
                raise ValueError(f"{directory}: state discontinuity at {index}")
            if not close(row["actuator_before"], prior["actuator_after"]):
                raise ValueError(f"{directory}: actuator discontinuity at {index}")
            if prior["fallback"] is None and not close(
                    row["command_queue_before"][0], prior["issued_command"]):
                raise ValueError(f"{directory}: command queue shift at {index}")
            if abs(row["ramped_nu_ref_mps"] - prior["ramped_nu_ref_mps"]) > .05000001:
                raise ValueError(f"{directory}: reference ramp at {index}")
        if row["fallback"] is None:
            executed += 1
            if row["selected_candidate_mps"] is None or row["candidate_empty"]:
                raise ValueError(f"{directory}: invalid selection at {index}")
        elif index != len(cycles) - 1:
            raise ValueError(f"{directory}: fallback without termination")
    if len(substeps) != 5 * executed or not finite(substeps):
        raise ValueError(f"{directory}: substep count or finite value mismatch")
    if any(substeps[i]["time_s"] <= substeps[i - 1]["time_s"]
           for i in range(1, len(substeps))):
        raise ValueError(f"{directory}: substep time not increasing")
    if summary["completed"] and (summary["abort_reason"] is not None or
                                 summary["backup_used"] or summary["candidate_empty"]):
        raise ValueError(f"{directory}: inconsistent completion flags")
    return {
        "directory": str(directory), "periods": len(cycles),
        "substeps": len(substeps), "completed": summary["completed"],
        "abort_reason": summary["abort_reason"],
        "first_physical_violation": summary["first_physical_violation"],
        "final_progress_m": summary["final_progress_m"],
        "path_length_m": summary["path_length_m"],
        "max_abs_lateral_m": summary["max_abs_lateral_m"],
        "max_abs_depth_m": summary["max_abs_depth_m"],
        "max_abs_speed_error_mps": summary["max_abs_speed_error_mps"],
        "mean_tangent_water_speed_mps": sum(
            row["actual_tangent_water_speed_mps"] for row in cycles) / len(cycles),
        "selected_intents": dict(Counter(str(row["selected_candidate_mps"])
                                         for row in cycles)),
        "rejections": dict(Counter(item["reason"] for row in cycles
                                   for item in row["candidate_rejections"]
                                   if item["reason"])),
        "p99_full_period_ms": summary["timing_ms"]["full_period"]["p99"],
        "max_full_period_ms": summary["timing_ms"]["full_period"]["max"],
        "state_after": [row["state_after"] for row in cycles],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for case, seed, prefix in RUNS:
        pair = {}
        for short, method in METHODS:
            directory = (ROOT / "outputs/experiments" /
                         f"gov_uuv_h15_smoke_s{seed}" /
                         f"{prefix}_{short}_{case}_20261104")
            pair[method] = audit_run(directory, method, case)
        pair["identical_states"] = (pair["B0_shared"]["state_after"] ==
                                    pair["B2_gov"]["state_after"])
        for method in ("B0_shared", "B2_gov"):
            del pair[method]["state_after"]
        results[case] = pair
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
