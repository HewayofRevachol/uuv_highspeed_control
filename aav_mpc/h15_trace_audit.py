"""Audit first H15 violations against frozen geometry and fin traces.

Reads completed source/target CSVs without changing either simulation. Fin
saturation and command/actuator lag are descriptive, not causal estimates.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_row(directory, family, seed, method, domain):
    suffix = "_target" if domain == "target" else ""
    stem = f"replication_{family}_s{seed}_{method}{suffix}"
    case_file = directory / f"{stem}.json"
    trace_file = directory / f"{stem}.csv"
    case = json.loads(case_file.read_text())
    with trace_file.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if (not rows or case.get("target_domain") != (domain == "target") or
            case.get("seed") != seed or case.get("method") != method or
            Path(case.get("path", "")).stem != f"replication_{family}"):
        raise ValueError(f"invalid {domain} episode: {case_file}")
    anomaly = case["first_anomaly"]
    cutoff = float(anomaly[0]) if anomaly else float(rows[-1]["time_s"])
    prefix = [row for row in rows if float(row["time_s"]) <= cutoff + 1e-8]
    if not prefix or abs(float(prefix[-1]["time_s"]) - cutoff) > 1e-6:
        raise ValueError(f"trace does not contain first anomaly: {trace_file}")
    saturated_elevator = [row for row in prefix if abs(float(row["elevator_actual_deg"])) >= 44.]
    saturated_rudder = [row for row in prefix if abs(float(row["rudder_actual_deg"])) >= 44.]
    tail = prefix[-20:]  # at most 2 s before first violation
    return {
        "family": family, "seed": seed, "method": method, "domain": domain,
        "strict_success": case["strict_success"],
        "first_anomaly_type": anomaly[1] if anomaly else None,
        "first_anomaly_time_s": cutoff if anomaly else None,
        "first_anomaly_progress_m": float(prefix[-1]["progress_m"]) if anomaly else None,
        "first_anomaly_pitch_deg": float(prefix[-1]["pitch_deg"]) if anomaly else None,
        "elevator_saturated_before_anomaly": bool(saturated_elevator),
        "rudder_saturated_before_anomaly": bool(saturated_rudder),
        "first_elevator_saturation_progress_m": (
            float(saturated_elevator[0]["progress_m"]) if saturated_elevator else None),
        "first_rudder_saturation_progress_m": (
            float(saturated_rudder[0]["progress_m"]) if saturated_rudder else None),
        "mean_abs_elevator_command_actuator_gap_last_2s_deg": sum(
            abs(float(row["elevator_command_deg"]) - float(row["elevator_actual_deg"]))
            for row in tail) / len(tail),
        "mean_abs_rudder_command_actuator_gap_last_2s_deg": sum(
            abs(float(row["rudder_command_deg"]) - float(row["rudder_actual_deg"]))
            for row in tail) / len(tail),
        "episode_sha256": digest(case_file), "trace_sha256": digest(trace_file),
    }


def summarize(rows, manifest):
    events = {p["family"]: p["geometry_events_m"]
              for p in manifest["splits"]["replication"]}
    for row in rows:
        progress = row["first_anomaly_progress_m"]
        row["first_anomaly_distance_after_nearest_event_m"] = (
            progress - min(events[row["family"]], key=lambda x: abs(progress-x))
            if progress is not None else None)
    result = {}
    for domain in ("source", "target"):
        selected = [row for row in rows if row["domain"] == domain]
        failed = [row for row in selected if not row["strict_success"]]
        result[domain] = {
            "episodes": len(selected),
            "strict_success": len(selected)-len(failed),
            "first_anomaly_types": dict(Counter(row["first_anomaly_type"] for row in failed)),
            "failed_first_progress_median_m": (
                median(row["first_anomaly_progress_m"] for row in failed) if failed else None),
            "failed_elevator_saturation_before_first": sum(
                row["elevator_saturated_before_anomaly"] for row in failed),
            "failed_rudder_saturation_before_first": sum(
                row["rudder_saturated_before_anomaly"] for row in failed),
            "failed_elevator_gap_last_2s_median_deg": (
                median(row["mean_abs_elevator_command_actuator_gap_last_2s_deg"]
                       for row in failed) if failed else None),
            "failed_rudder_gap_last_2s_median_deg": (
                median(row["mean_abs_rudder_command_actuator_gap_last_2s_deg"]
                       for row in failed) if failed else None),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/datasets/h15_mpc_replication/manifest.json"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("endpoint20", "condensed_endpoint35",
                                            "condensed_endpoint40"),
                        default="condensed_endpoint35")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    target_run = json.loads((args.target_dir / "run.json").read_text())
    if (target_run.get("source_run_sha256") != digest(args.source_dir / "run.json") or
            target_run.get("source_dir") != str(args.source_dir) or
            target_run.get("corpus") != "replication" or
            target_run.get("split") != "replication" or
            args.method not in target_run.get("methods", [])):
        parser.error("source and target runs are not the selected paired replication")
    rows = []
    for path in manifest["splits"]["replication"]:
        if path["generator_family"] != "coupled":
            continue
        for seed in manifest["experiment_freeze"]["seeds"]:
            for domain, directory in (("source", args.source_dir),
                                      ("target", args.target_dir)):
                row = trace_row(directory, path["family"], seed, args.method, domain)
                rows.append(row)
    output = {"method": args.method, "manifest_sha256": digest(args.manifest),
              "source_run_sha256": digest(args.source_dir / "run.json"),
              "target_run_sha256": digest(args.target_dir / "run.json"),
              "summary": summarize(rows, manifest), "episodes": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
