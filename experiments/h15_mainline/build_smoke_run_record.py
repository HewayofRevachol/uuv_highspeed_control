"""Apply the shared run.json schema to profiled reference-chain smoke data."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.h15_mainline.run_record import (TOP_LEVEL, TIMING_STAGES,
                                                  RESULT_FIELDS,
                                                  write_once)
from experiments.h15_mainline.cycle_log import (FIELDS as CYCLE_FIELDS,
                                                TIMING_FIELDS as CYCLE_TIMING_FIELDS,
                                                validate as validate_cycle)
from experiments.h15_mainline.strict_contract import first_physical_violation
from experiments.h15_mainline.cycle_writer import (timing_summary,
                                                   write_once as write_cycles_once)


ROOT = Path(__file__).resolve().parents[2]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    summaries = json.loads((args.smoke / "summary.json").read_text())
    if len(summaries) != 6 or any(not item["profile_stages"] for item in summaries):
        raise RuntimeError("expected six profiled smoke episodes")
    smoke_run = json.loads((args.smoke / "run.json").read_text())
    if smoke_run["freeze_sha256"] != sha(args.freeze):
        raise RuntimeError("smoke run references another freeze")
    configs = {"quick_run_manifest.json": sha(args.freeze),
               "task_contract.json": sha(ROOT / "experiments/h15_mainline/task_contract.json"),
               "h15.json": sha(ROOT / "aav_mpc/configs/h15.json"),
               "run_record.py": sha(ROOT / "experiments/h15_mainline/run_record.py"),
               "cycle_log.py": sha(ROOT / "experiments/h15_mainline/cycle_log.py"),
               "cycle_writer.py": sha(ROOT / "experiments/h15_mainline/cycle_writer.py")}
    data = {f"{split}_manifest.json": item["sha256"]
            for split, item in freeze["manifests"].items()}
    first = Counter()
    cycles = []
    for item in summaries:
        trace = ROOT / item["trace"]
        data[item["trace"]] = sha(trace)
        with trace.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        substep_trace = trace.with_name(trace.stem + "_substeps.csv")
        with substep_trace.open(newline="") as stream:
            substeps = list(csv.DictReader(stream))
        if len(substeps) != 5 * len(rows):
            raise RuntimeError(f"incomplete substep trace: {substep_trace}")
        first_event = first_physical_violation(substeps)
        first_period = None
        if first_event is not None:
            first_period = next(int(row["period"]) for row in substeps
                                if abs(float(row["time_s"])-first_event[0]) < 1e-9)
        if len(rows) != item["steps"]:
            raise RuntimeError(f"incomplete trace: {trace}")
        if item["first_anomaly"] is not None:
            first[item["first_anomaly"][1]] += 1
        for period, row in enumerate(rows):
            actual_after = [float(row["thrust_actual_n"]),
                            math.radians(float(row["elevator_actual_deg"])),
                            math.radians(float(row["rudder_actual_deg"]))]
            cycle = dict(zip(CYCLE_FIELDS, (
                1, "reference_chain_smoke", trace.stem, period,
                float(row["time_s"]), json.loads(row["state_before_json"]),
                json.loads(row["state_after_json"]),
                json.loads(row["actuator_before_json"]), actual_after,
                json.loads(row["queue_before_json"]),
                json.loads(row["command_json"]), None,
                float(row["nu_ref_mps"]),
                float(row["actual_tangent_water_speed_mps"]),
                None, None, None,
                None if row["mpc_ok"] == "True" else "mpc_fallback",
                ({"time_s": first_event[0], "kind": first_event[1]}
                 if period == first_period else None),
                row["mpc_status"],
                dict(zip(CYCLE_TIMING_FIELDS, (
                    None, None, None, float(row["mpc_wall_ms"]),
                    float(row["plant_integration_wall_ms"]),
                    float(row["control_period_wall_ms"])))))))
            cycles.append(validate_cycle(cycle))
    timings = timing_summary(cycles)
    if tuple(timings) != TIMING_STAGES:
        raise RuntimeError("cycle and run timing stage schemas differ")
    results = dict.fromkeys(RESULT_FIELDS)
    results.update(episodes=len(summaries),
                   strict_hist_completed=sum(bool(item["strict_hist"]) for item in summaries),
                   strict_mainline_completed=None,
                   completed_with_backup=None,
                   fallback_episodes=sum(item["fallback_count"] > 0 for item in summaries),
                   empty_candidate_episodes=None,
                   first_violations=dict(first))
    record = dict(zip(TOP_LEVEL, (1, "h15_reference_chain_smoke_profiled",
                                  configs, data,
                                  sorted({item["seed"] for item in summaries}),
                                  timings, results)))
    args.out.mkdir(parents=True, exist_ok=True)
    write_once(args.out / "run.json", record)
    write_cycles_once(args.out / "cycle_log.jsonl", cycles)
    (args.out / "console.log").write_text(
        f"validated {len(cycles)} diagnostic cycles; candidate, policy and strict_mainline unavailable\n")
    print(f"validated {len(cycles)} diagnostic cycles in {len(summaries)} smoke episodes")


if __name__ == "__main__":
    main()
