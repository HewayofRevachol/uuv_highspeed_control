"""Generate smooth, arclength sampled 15 m/s diagnostic geometries.

The split is by geometry parameters, never by changing the speed of a CSV.
Each path is deterministic and has a 60 m straight entry for actuator spin-up.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


FAMILIES = {
    "turn": ("constant_turn", "radius_m"),
    "s": ("s_bend", "peak_curvature_per_m"),
    "depth": ("depth_wave", "amplitude_m"),
    "entry": ("varying_entry_turn", "entry_m"),
}
SPLITS = {
    "development": {"turn": 180., "s": .0060, "depth": 8., "entry": 75.},
    "validation": {"turn": 220., "s": .0050, "depth": 6., "entry": 95.},
    "test": {"turn": 155., "s": .0065, "depth": 10., "entry": 65.},
}


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3. - 2.*x)


def generate(family, parameter, length_m=240., ds_m=.5):
    s = np.linspace(0., length_m, round(length_m / ds_m) + 1)
    kappa = np.zeros_like(s)
    z = np.full_like(s, 280.)
    if family == "turn":
        kappa = smoothstep((s - 60.) / 35.) / parameter
    elif family == "s":
        kappa = parameter * (smoothstep((s - 60.) / 25.) -
                             2.*smoothstep((s - 125.) / 30.))
    elif family == "depth":
        phase = np.clip((s - 60.) / 180., 0., 1.)
        z += parameter * np.sin(2. * np.pi * phase) * smoothstep((s - 60.) / 25.)
    elif family == "entry":
        kappa = smoothstep((s - parameter) / 30.) / 175.
    else:
        raise ValueError(f"unknown family: {family}")
    # Integrate the unit tangent at midpoints so the CSV's geometric arc
    # length differs from the design arclength only at O(ds^2).
    yaw = np.r_[0., np.cumsum((kappa[:-1] + kappa[1:]) * np.diff(s) / 2.)]
    dz = np.gradient(z, s)
    horizontal = np.sqrt(1. - np.minimum(dz*dz, .99))
    tangent = np.column_stack((horizontal*np.cos(yaw), horizontal*np.sin(yaw), dz))
    midpoint = (tangent[:-1] + tangent[1:]) / 2.
    points = np.vstack((np.array([0., 0., 280.]),
                        np.array([0., 0., 280.]) + np.cumsum(midpoint*np.diff(s)[:, None], axis=0)))
    # Integrating dz restores the chosen depth profile exactly up to roundoff.
    points[:, 2] = z
    if not np.isfinite(points).all() or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0):
        raise ValueError("invalid generated path")
    return points


def write_corpus(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "speed_mps": 15., "length_design_m": 240.,
                "sample_spacing_m": .5, "splits": {}, "families": FAMILIES}
    for split, settings in SPLITS.items():
        manifest["splits"][split] = []
        for family, value in settings.items():
            file = directory / f"{split}_{family}.csv"
            points = generate(family, value)
            with file.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("north_m", "east_m", "depth_m", "target_tangent_speed_mps"))
                writer.writerows((*map(float, point), 15.) for point in points)
            manifest["splits"][split].append({"family": family, "parameter": value,
                                                "path": str(file)})
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/datasets/h15_mpc"))
    args = parser.parse_args()
    write_corpus(args.out)
