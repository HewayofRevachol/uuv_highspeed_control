"""Prospective H15 replication geometries after the initial holdout diagnosis.

Four coupled variants separate curvature, wave amplitude, wave length, and
entry timing. Two non-coupled paths check that a fix does not break easier
families. Parameters and seeds are frozen here before running controllers.
"""
import argparse
import csv
import json
from pathlib import Path

from aav_mpc.h15_challenge_paths import generate


PROFILES = (
    ("tight_turn_rep", "tight_turn", {"radius_m": 47., "entry_m": 16.}),
    ("reversal_rep", "reversal", {"radius_m": 63., "reversal_m": 150., "transition_m": 22.}),
    ("coupled_a", "coupled", {"radius_m": 65., "depth_amplitude_m": 8.8,
                              "start_m": 60., "entry_m": 20., "wave_length_m": 180.}),
    ("coupled_b", "coupled", {"radius_m": 70., "depth_amplitude_m": 9.3,
                              "start_m": 55., "entry_m": 18., "wave_length_m": 165.}),
    ("coupled_c", "coupled", {"radius_m": 60., "depth_amplitude_m": 8.2,
                              "start_m": 70., "entry_m": 22., "wave_length_m": 190.}),
    ("coupled_d", "coupled", {"radius_m": 68., "depth_amplitude_m": 8.9,
                              "start_m": 50., "entry_m": 16., "wave_length_m": 200.}),
)
METHODS = ("endpoint20", "condensed_endpoint35", "condensed_endpoint40")
SEEDS = tuple(range(101041, 101051))


def write_corpus(directory):
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, generator_family, parameters in PROFILES:
        points, events = generate(generator_family, parameters)
        file = directory / f"replication_{name}.csv"
        with file.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("north_m", "east_m", "depth_m", "target_tangent_speed_mps"))
            writer.writerows((*map(float, point), 15.) for point in points)
        entries.append({"family": name, "generator_family": generator_family,
                        "parameters": parameters, "geometry_events_m": events,
                        "path": str(file)})
    manifest = {"schema": 1, "speed_mps": 15., "length_design_m": 300.,
                "sample_spacing_m": .5,
                "experiment_freeze": {"methods": METHODS, "seeds": SEEDS,
                                      "task_config": "aav_mpc/configs/h15.json"},
                "splits": {"replication": entries}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("data/datasets/h15_mpc_replication"))
    args = parser.parse_args()
    write_corpus(args.out)
