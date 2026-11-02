"""Generate original, arc-length-parametrized DDAT training and validation paths.

The five maneuver families are built from smooth curvature, depth, and speed
profiles.  This generator does not read or transform any existing trajectory.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SPACING_M = 0.8


def smootherstep(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(x*(6*x-15)+10)


@dataclass(frozen=True)
class Recipe:
    name: str
    length_m: float
    initial_depth_m: float
    initial_speed_mps: float
    bends: tuple = ()       # (start_m, end_m, ramp_m, signed_curvature_1pm)
    depth_moves: tuple = () # (start_m, span_m, signed_depth_change_m)
    speed_moves: tuple = () # (start_m, span_m, signed_speed_change_mps)


def build_recipe(family, side, radius, speed_peak, phase, depth_change=0.):
    """Construct a case from physical maneuver parameters, not old path data."""
    sign = float(side)
    k = sign/radius if radius else 0.
    if family == "speed_envelope":
        return Recipe(family, 380.+phase, 280., 15.3,
            speed_moves=((45., 55.+phase/8, speed_peak-15.3),
                         (180.+phase/2, 55., -(speed_peak-15.3)),
                         (285.+phase/2, 50., speed_peak-15.3)))
    if family == "constant_turn":
        return Recipe(family, 360.+phase, 280., 15.6,
            bends=((90., 240.+phase/2, 20., k),),
            speed_moves=((70., 50., speed_peak-15.6),
                         (260.+phase/2, 48., -(speed_peak-15.6))))
    if family == "s_reversal":
        return Recipe(family, 410.+phase, 280., 15.5,
            bends=((75., 165.+phase/3, 18., k),
                   (164.+phase/3, 275.+phase/2, 18., -0.92*k),
                   (275.+phase/2, 340.+phase/2, 18., 0.36*k)),
            speed_moves=((55., 48., speed_peak-15.5),
                         (295.+phase/2, 55., -(speed_peak-15.5))))
    if family == "depth_return":
        return Recipe(family, 410.+phase, 280., 15.6,
            depth_moves=((85., 90.+phase/5, depth_change),
                         (232.+phase/2, 90., -depth_change)),
            speed_moves=((65., 70., speed_peak-15.6),
                         (265.+phase/2, 65., -(speed_peak-15.6))))
    if family == "coupled_axes":
        return Recipe(family, 460.+phase, 280., 15.5,
            bends=((88., 185.+phase/3, 20., k),
                   (205.+phase/3, 330.+phase/2, 20., -0.82*k)),
            depth_moves=((104., 95.+phase/5, depth_change),
                         (250.+phase/2, 95., -depth_change)),
            speed_moves=((68., 50., speed_peak-15.5),
                         (178.+phase/3, 70., -(speed_peak-15.5)),
                         (282.+phase/2, 45., 0.9)))
    raise ValueError(f"unknown maneuver family: {family}")


# Each validation row is a held-out combination, separate from the training
# parameter grid.  The two turn directions and both depth signs are trained.
TRAIN = {
    "speed_envelope": ((1, 0., 17.8, -16.), (-1, 0., 18.6, 8.),
                       (1, 0., 19.0, 20.), (-1, 0., 19.4, 30.)),
    "constant_turn": ((1, 88., 16.6, -12.), (-1, 65., 17.1, 6.),
                      (1, 52., 17.6, 18.), (-1, 45., 18.2, 30.)),
    "s_reversal": ((1, 85., 16.6, -12.), (-1, 60., 17.2, 6.),
                   (1, 50., 17.7, 18.), (-1, 42., 18.2, 30.)),
    "depth_return": ((1, 0., 16.7, -12., 15.),
                     (-1, 0., 17.2, 6., -19.),
                     (1, 0., 17.6, 18., 21.),
                     (-1, 0., 18.0, 30., -23.)),
    "coupled_axes": ((1, 86., 16.8, -12., 12.),
                     (-1, 68., 17.2, 6., -17.),
                     (1, 55., 17.6, 18., -21.),
                     (-1, 48., 18.0, 30., 23.)),
}

VAL = {
    "speed_envelope": ((1, 0., 19.2, 14.),),
    "constant_turn": ((-1, 48., 18.0, 13.),),
    "s_reversal": ((1, 45., 18.0, 13.),),
    "depth_return": ((-1, 0., 17.5, 13., 22.),),
    "coupled_axes": ((-1, 52., 17.4, 13., 22.),),
}


def sample(recipe):
    count = int(np.ceil(recipe.length_m/SPACING_M))+1
    s = np.linspace(0., recipe.length_m, count)
    curvature = np.zeros_like(s)
    for start, end, ramp, peak in recipe.bends:
        curvature += peak*smootherstep((s-start)/ramp)*smootherstep((end-s)/ramp)
    heading = np.zeros_like(s)
    heading[1:] = np.cumsum((curvature[:-1]+curvature[1:])*np.diff(s)/2)
    north = np.zeros_like(s)
    east = np.zeros_like(s)
    north[1:] = np.cumsum((np.cos(heading[:-1])+np.cos(heading[1:]))*np.diff(s)/2)
    east[1:] = np.cumsum((np.sin(heading[:-1])+np.sin(heading[1:]))*np.diff(s)/2)
    depth = np.full_like(s, recipe.initial_depth_m)
    for start, span, change in recipe.depth_moves:
        depth += change*smootherstep((s-start)/span)
    speed = np.full_like(s, recipe.initial_speed_mps)
    for start, span, change in recipe.speed_moves:
        speed += change*smootherstep((s-start)/span)
    return np.column_stack((north, east, depth, speed)), s, curvature


def write_set(root, split, cases):
    destination = root/split
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    for family, variants in cases.items():
        for index, values in enumerate(variants, 1):
            recipe = build_recipe(family, *values)
            data, distance, curvature = sample(recipe)
            positions = data[:, :3]
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
            tangent = np.gradient(positions, arc, axis=0)
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            pitch = -np.arctan2(tangent[:, 2], np.linalg.norm(tangent[:, :2], axis=1))
            yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
            file = destination/f"{family}_{index:02d}.csv"
            with file.open("w", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(("north_m", "east_m", "depth_m", "target_pitch_rad",
                                 "target_yaw_rad", "target_tangent_speed_mps"))
                writer.writerows((f"{x:.9f}" for x in row) for row in
                                 np.column_stack((positions, pitch, yaw, data[:, 3])))
            # Values below describe the generated geometry, independent of
            # vehicle dynamics or any controller outcome.
            arc_length = np.linalg.norm(np.diff(data[:, :3], axis=0), axis=1).sum()
            summary.append({"path": str(file.relative_to(root)), "family": family,
                            "length_m": round(float(arc_length), 2),
                            "min_speed_mps": round(float(data[:, 3].min()), 2),
                            "max_speed_mps": round(float(data[:, 3].max()), 2),
                            "min_depth_m": round(float(data[:, 2].min()), 2),
                            "max_depth_m": round(float(data[:, 2].max()), 2),
                            "peak_abs_curvature_1pm": round(float(np.abs(curvature).max()), 5)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    records = write_set(args.output, "train", TRAIN)+write_set(args.output, "val", VAL)
    (args.output/"manifest.json").write_text(json.dumps(records, indent=2)+"\n")
    print(f"Wrote {len(records)} paths to {args.output}")


if __name__ == "__main__":
    main()
