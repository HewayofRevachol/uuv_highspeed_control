"""Generate a reproducible family of smooth turns for RL reference control.

Training radii and entry lengths deliberately exclude the original 40 m,
15 m-entry pressure-test helix used for held-out evaluation.
"""

import argparse
from pathlib import Path

import numpy as np


HEADER = "north_m,east_m,depth_m,target_tangent_speed_mps"


def smoothstep(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3. - 2. * value)


def make_turn(radius, entry_m, depth_slope, speed=18.):
    length = 4. * np.pi * radius + 50.
    distance = np.linspace(0., length, int(np.ceil(length / .4)) + 1)
    curvature = smoothstep(distance / entry_m) / radius
    step = np.diff(distance)
    heading = np.r_[0., np.cumsum((curvature[:-1] + curvature[1:]) * step / 2.)]
    midpoint = .5 * (heading[:-1] + heading[1:])
    north = np.r_[0., np.cumsum(step * np.cos(midpoint))]
    east = np.r_[0., np.cumsum(step * np.sin(midpoint))]
    depth = 280. + depth_slope * distance
    target = np.full_like(distance, speed)
    return np.column_stack((north, east, depth, target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = []
    for radius in (36., 44., 52., 60.):
        for entry in (10., 20., 30.):
            for slope in (0., .04):
                file = args.output / f"turn_r{int(radius)}_entry{int(entry)}_slope{int(100*slope):02d}.csv"
                np.savetxt(file, make_turn(radius, entry, slope), delimiter=",",
                           fmt="%.9f", header=HEADER, comments="")
                files.append(str(file))
    print(f"generated {len(files)} training paths in {args.output}")


if __name__ == "__main__":
    main()
