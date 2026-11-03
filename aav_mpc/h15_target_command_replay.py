"""Replay source-feasible commands on a fixed target-style plant.

This is a cheap transfer screen. A failed replay does not prove target-path
infeasibility because commands were optimized for the source actuator.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from aav_mpc.offline_feasibility import replay
from simulator.model import LoopPath, MISMATCHED_PLANT


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_violation(trace, task):
    limits = (("lateral_m", task["lateral_corridor_m"]),
              ("depth_m", task["depth_corridor_m"]),
              ("pitch_deg", task["pitch_limit_deg"]),
              ("speed_error_mps", task["speed_tolerance_mps"]))
    for row in trace:
        for name, limit in limits:
            if abs(row[name]) > limit:
                return {"time_s": row["time_s"], "progress_m": row["progress_m"],
                        "type": name, "value": row[name], "limit": limit}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/datasets/h15_mpc_replication/manifest.json"))
    parser.add_argument("--source-results-dir", type=Path,
                        default=Path("outputs/experiments"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    plant = replace(MISMATCHED_PLANT,
                    elevator_gain=.9 * MISMATCHED_PLANT.elevator_gain,
                    rudder_gain=.9 * MISMATCHED_PLANT.rudder_gain)
    rows = []
    for entry in manifest["splits"]["replication"]:
        if entry["generator_family"] != "coupled":
            continue
        stem = f"mpc_h15_replication_{entry['family']}_offline_feasibility"
        source_file = args.source_results_dir / f"{stem}.json"
        command_file = args.source_results_dir / f"{stem}_start0_commands.npy"
        source = json.loads(source_file.read_text())
        if (source["task"]["path"] != entry["path"] or
                not source["results"][0]["found_full_path_feasible"] or
                source["results"][0]["solver_status"] != "solved"):
            raise ValueError(f"source candidate is not a verified full-path solution: {source_file}")
        commands = np.load(command_file, allow_pickle=False)
        if commands.shape != (round(source["horizon_s"] / source["block_s"]), 3):
            raise ValueError(f"invalid source command shape: {command_file}")
        task = dict(source["task"])
        path = LoopPath(entry["path"])
        source_replay = replay(path, task, commands, source["block_s"])
        target_replay = replay(LoopPath(entry["path"]), task, commands,
                               source["block_s"], plant=plant, target_domain=True)
        source_first = first_violation(source_replay["trace"], task)
        target_first = first_violation(target_replay["trace"], task)
        source_replay.pop("trace")
        target_replay.pop("trace")
        rows.append({"family": entry["family"], "path": entry["path"],
                     "source_result_sha256": sha256(source_file),
                     "commands_sha256": sha256(command_file),
                     "source_replay": {**source_replay, "first_violation": source_first},
                     "fixed_target_replay": {**target_replay, "first_violation": target_first}})
    output = {"manifest_sha256": sha256(args.manifest), "source_results_dir":
              str(args.source_results_dir), "target_plant": vars(plant),
              "target_integrator": "two_substep_midpoint",
              "target_actuator": "critical_second_order",
              "current_max_mps": 0., "process_noise": False, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["family"], "source", row["source_replay"]["found_full_path_feasible"],
              "target", row["fixed_target_replay"]["found_full_path_feasible"],
              "target first", row["fixed_target_replay"]["first_violation"])


if __name__ == "__main__":
    main()
