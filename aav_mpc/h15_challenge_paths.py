"""Near-envelope H15 maneuvers with explicit geometric events.

Radii 45-70 m are chosen after the fixed-mismatch plant's 15 m/s steady
trim check: R30 needs 1.64 MN (above the 1.5 MN bound), whereas R40 needs
1.35 MN. Transient feasibility is an empirical question, not assumed here.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from aav_mpc.h15_paths import smoothstep


PROFILES = {
    "development": {
        "tight_turn": {"radius_m": 55., "entry_m": 20.},
        "reversal": {"radius_m": 65., "reversal_m": 155., "transition_m": 25.},
        "coupled": {"radius_m": 70., "depth_amplitude_m": 8.},
        "late_entry": {"radius_m": 45., "entry_m": 105., "transition_m": 12.},
    },
    "validation": {
        "tight_turn": {"radius_m": 60., "entry_m": 25.},
        "reversal": {"radius_m": 70., "reversal_m": 170., "transition_m": 30.},
        "coupled": {"radius_m": 75., "depth_amplitude_m": 7.},
        "late_entry": {"radius_m": 50., "entry_m": 115., "transition_m": 16.},
    },
    "test": {
        "tight_turn": {"radius_m": 50., "entry_m": 18.},
        "reversal": {"radius_m": 60., "reversal_m": 145., "transition_m": 20.},
        "coupled": {"radius_m": 62., "depth_amplitude_m": 9.},
        "late_entry": {"radius_m": 42., "entry_m": 95., "transition_m": 10.},
    },
}


def generate(family, parameters, length_m=300., ds_m=.5):
    s = np.linspace(0., length_m, round(length_m / ds_m) + 1)
    radius = parameters["radius_m"]
    z = np.full_like(s, 280.)
    if family == "tight_turn":
        start = 60.
        kappa = smoothstep((s - start) / parameters["entry_m"]) / radius
        events = [start, start + parameters["entry_m"]]
    elif family == "reversal":
        start, flip, width = 60., parameters["reversal_m"], parameters["transition_m"]
        kappa = (smoothstep((s-start)/20.) - 2.*smoothstep((s-flip)/width)) / radius
        events = [start, start+20., flip, flip+width]
    elif family == "coupled":
        start = parameters.get("start_m", 60.)
        entry = parameters.get("entry_m", 20.)
        wave_length = parameters.get("wave_length_m", 180.)
        kappa = smoothstep((s-start)/entry) / radius
        phase = np.clip((s-start)/wave_length, 0., 1.)
        z += parameters["depth_amplitude_m"] * np.sin(2.*np.pi*phase) * smoothstep((s-start)/entry)
        events = [start, start+entry, start+wave_length]
    elif family == "late_entry":
        start, width = parameters["entry_m"], parameters["transition_m"]
        kappa = smoothstep((s-start)/width) / radius
        events = [start, start+width]
    else:
        raise ValueError(family)
    yaw = np.r_[0., np.cumsum((kappa[:-1]+kappa[1:])*np.diff(s)/2.)]
    dz = np.gradient(z, s)
    if np.max(abs(dz)) >= .8:
        raise ValueError("depth slope exceeds design envelope")
    horizontal = np.sqrt(1.-dz*dz)
    tangent = np.column_stack((horizontal*np.cos(yaw), horizontal*np.sin(yaw), dz))
    midpoint = (tangent[:-1]+tangent[1:])/2.
    points = np.vstack((np.array([0., 0., 280.]),
                        np.array([0., 0., 280.])+np.cumsum(midpoint*np.diff(s)[:, None], axis=0)))
    points[:, 2] = z
    return points, events


def write_corpus(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "speed_mps": 15., "length_design_m": 300.,
                "sample_spacing_m": .5, "splits": {}}
    for split, profiles in PROFILES.items():
        manifest["splits"][split] = []
        for family, parameters in profiles.items():
            points, events = generate(family, parameters)
            file = directory / f"{split}_{family}.csv"
            with file.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("north_m", "east_m", "depth_m", "target_tangent_speed_mps"))
                writer.writerows((*map(float, point), 15.) for point in points)
            manifest["splits"][split].append({"family": family, "parameters": parameters,
                                                "geometry_events_m": events, "path": str(file)})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/datasets/h15_mpc_challenge"))
    args = parser.parse_args()
    write_corpus(args.out)
