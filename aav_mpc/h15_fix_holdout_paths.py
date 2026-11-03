"""Freeze prospective H15 geometries for evaluating a deterministic MPC fix."""
import argparse
import csv
import json
from pathlib import Path

from aav_mpc.h15_challenge_paths import generate


PROFILES = (
    ("holdout_coupled_e", {"radius_m": 58., "depth_amplitude_m": 8.5,
                           "start_m": 52., "entry_m": 14., "wave_length_m": 175.}),
    ("holdout_coupled_f", {"radius_m": 72., "depth_amplitude_m": 9.0,
                           "start_m": 66., "entry_m": 24., "wave_length_m": 210.}),
    ("holdout_coupled_g", {"radius_m": 64., "depth_amplitude_m": 9.4,
                           "start_m": 58., "entry_m": 19., "wave_length_m": 160.}),
    ("holdout_coupled_h", {"radius_m": 66., "depth_amplitude_m": 8.6,
                           "start_m": 74., "entry_m": 17., "wave_length_m": 195.}),
)
METHODS = ("endpoint20", "condensed_endpoint35", "robust_condensed_endpoint35")
SEEDS = tuple(range(102061, 102066))


def write_corpus(directory):
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for family, parameters in PROFILES:
        points, events = generate("coupled", parameters)
        file = directory / f"{family}.csv"
        with file.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("north_m", "east_m", "depth_m", "target_tangent_speed_mps"))
            writer.writerows((*map(float, point), 15.) for point in points)
        entries.append({"family": family, "generator_family": "coupled",
                        "parameters": parameters, "geometry_events_m": events,
                        "path": str(file)})
    manifest = {
        "schema": 1,
        "speed_mps": 15.,
        "length_design_m": 300.,
        "sample_spacing_m": .5,
        "experiment_freeze": {
            "methods": METHODS,
            "seeds": SEEDS,
            "task_config": "aav_mpc/configs/h15.json",
            "target_config": "aav_mpc/configs/h15_target.json",
            "robust_prediction": {"fin_gain_scale": .85, "fin_tau_scale": 2.},
        },
        "splits": {"holdout": entries},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("data/datasets/h15_mpc_fix_holdout"))
    args = parser.parse_args()
    print(json.dumps(write_corpus(args.out), indent=2))
