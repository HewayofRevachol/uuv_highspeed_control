"""Generate varied path CSVs and local path windows for encoder pretraining.

The corpus contains only reference paths; generating it does not run the UUV
simulation or MPC. Splits are assigned per path before windows are extracted.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.model import LoopPath


FAMILIES = (
    "straight", "single_turn", "s_turn", "slalom", "depth_wave",
    "coupled_turn_depth", "helix", "speed_change", "mixed",
)
PATH_COLUMNS = "north_m,east_m,depth_m,target_tangent_speed_mps"
WINDOW_FEATURES = (
    "forward_50m", "lateral_50m", "depth_50m", "yaw_pi",
    "pitch_0p5rad", "target_speed_15mps", "valid",
)


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def turn_window(distance, start, ramp, hold):
    return (smoothstep((distance - start) / ramp) *
            (1.0 - smoothstep((distance - start - ramp - hold) / ramp)))


def make_path(family, rng, spacing_m):
    """Return a four-column trajectory and its generation parameters."""
    length = float(rng.uniform(800.0, 1200.0))
    helix_params = None
    if family == "helix":
        radius = float(rng.uniform(55.0, 130.0))
        turns = float(rng.uniform(0.8, 1.5))
        entry = float(rng.uniform(30.0, 80.0))
        ramp = float(rng.uniform(40.0, 90.0))
        tail = float(rng.uniform(50.0, 120.0))
        hold = 2.0 * np.pi * radius * turns
        length = entry + 2.0 * ramp + hold + tail
        helix_params = (radius, turns, entry, ramp, hold)

    distance = np.linspace(0.0, length,
                           int(np.ceil(length / spacing_m)) + 1)
    curvature = np.zeros_like(distance)
    depth = np.full_like(distance, 280.0)
    base_speed = float(rng.uniform(14.0, 16.0))
    speed = np.full_like(distance, base_speed)
    parameters = {"horizontal_length_m": length, "base_speed_mps": base_speed,
                  "turns": [], "depth_changes": [], "speed_changes": []}

    def add_turn(start, ramp, hold, radius, sign):
        curvature[:] += sign * turn_window(distance, start, ramp, hold) / radius
        parameters["turns"].append({
            "start_m": float(start), "ramp_m": float(ramp),
            "hold_m": float(hold), "radius_m": float(radius),
            "direction": int(sign),
        })

    def add_depth(start, duration, change):
        depth[:] += change * smoothstep((distance - start) / duration)
        parameters["depth_changes"].append({
            "start_m": float(start), "duration_m": float(duration),
            "change_m": float(change),
        })

    def add_speed(start, duration, target):
        current = float(np.interp(start, distance, speed))
        speed[:] += (target - current) * smoothstep((distance - start) / duration)
        parameters["speed_changes"].append({
            "start_m": float(start), "duration_m": float(duration),
            "target_mps": float(target),
        })

    sign = int(rng.choice((-1, 1)))
    if family == "single_turn":
        add_turn(rng.uniform(.12, .30) * length,
                 rng.uniform(.06, .12) * length,
                 rng.uniform(.14, .25) * length,
                 rng.uniform(90.0, 240.0), sign)
    elif family in ("s_turn", "coupled_turn_depth"):
        radius_range = (50.0, 110.0) if family == "coupled_turn_depth" else (75.0, 180.0)
        for index, start_fraction in enumerate((.12, .52)):
            add_turn(start_fraction * length, rng.uniform(.045, .075) * length,
                     rng.uniform(.13, .19) * length,
                     rng.uniform(*radius_range), sign * (-1 if index else 1))
        if family == "coupled_turn_depth":
            amplitude = float(rng.uniform(15.0, 35.0)) * int(rng.choice((-1, 1)))
            for start_fraction, change in ((.10, amplitude),
                                           (.42, -2.0 * amplitude),
                                           (.72, amplitude)):
                add_depth(start_fraction * length,
                          rng.uniform(.12, .17) * length, change)
    elif family == "slalom":
        count = int(rng.integers(3, 5))
        for index in range(count):
            add_turn((.08 + .20 * index) * length,
                     rng.uniform(.025, .04) * length,
                     rng.uniform(.07, .10) * length,
                     rng.uniform(65.0, 150.0), sign * (-1) ** index)
    elif family == "depth_wave":
        amplitude = float(rng.uniform(12.0, 35.0)) * sign
        for start_fraction, change in ((.13, amplitude),
                                       (.43, -2.0 * amplitude),
                                       (.73, amplitude)):
            add_depth(start_fraction * length,
                      rng.uniform(.12, .18) * length, change)
    elif family == "helix":
        radius, turns, entry, ramp, hold = helix_params
        add_turn(entry, ramp, hold, radius, sign)
        slope = float(rng.uniform(.015, .06)) * int(rng.choice((-1, 1)))
        depth += slope * distance
        parameters["helix_turns"] = turns
        parameters["depth_slope_m_per_m"] = slope
    elif family == "speed_change":
        for fraction in (.18, .48, .73):
            add_speed(fraction * length, rng.uniform(.07, .12) * length,
                      rng.uniform(14.0, 16.0))
    elif family == "mixed":
        count = int(rng.integers(1, 4))
        for index in range(count):
            add_turn((.12 + .27 * index) * length,
                     rng.uniform(.035, .055) * length,
                     rng.uniform(.09, .13) * length,
                     rng.uniform(70.0, 200.0),
                     int(rng.choice((-1, 1))))
        amplitude = float(rng.uniform(8.0, 25.0)) * sign
        add_depth(.20 * length, .16 * length, amplitude)
        add_depth(.62 * length, .18 * length, -amplitude)
        add_speed(.45 * length, .10 * length, rng.uniform(14.0, 16.0))
    elif family != "straight":
        raise ValueError(f"unknown path family: {family}")

    if family in ("single_turn", "s_turn", "slalom", "depth_wave",
                  "coupled_turn_depth", "helix") and rng.random() < .35:
        add_speed(.55 * length, .10 * length, rng.uniform(14.0, 16.0))

    increments = np.diff(distance)
    heading = np.r_[0.0, np.cumsum(
        .5 * (curvature[:-1] + curvature[1:]) * increments)]
    mid_heading = .5 * (heading[:-1] + heading[1:])
    north = np.r_[0.0, np.cumsum(increments * np.cos(mid_heading))]
    east = np.r_[0.0, np.cumsum(increments * np.sin(mid_heading))]
    columns = np.column_stack((north, east, depth, speed))
    if not np.isfinite(columns).all() or np.any(speed <= 0):
        raise RuntimeError(f"invalid generated trajectory: {family}")
    parameters["max_abs_horizontal_curvature_per_m"] = float(
        np.max(np.abs(curvature)))
    return columns, parameters


def extract_windows(path, start_step_m, preview_step_m, preview_points):
    """Return path-relative, normalized windows with shape (N, K, 7)."""
    starts = np.arange(0.0, path.length, start_step_m)
    offsets = preview_step_m * np.arange(preview_points)
    requested = starts[:, None] + offsets[None, :]
    valid = requested <= path.length
    sampled = np.minimum(requested, path.length)
    origin = np.stack([np.interp(starts, path.distances, axis)
                       for axis in path.points.T], axis=-1)
    position = np.stack([np.interp(sampled, path.distances, axis)
                         for axis in path.points.T], axis=-1)
    delta = position - origin[:, None, :]
    yaw0 = np.interp(starts, path.distances, path.yaw)[:, None]
    pitch0 = np.interp(starts, path.distances, path.pitch)[:, None]
    yaw = np.interp(sampled, path.distances, path.yaw)
    pitch = np.interp(sampled, path.distances, path.pitch)
    speed = np.interp(sampled, path.distances, path.speed)
    relative_yaw = yaw - yaw0
    windows = np.stack((
        (np.cos(yaw0) * delta[..., 0] + np.sin(yaw0) * delta[..., 1]) / 50.0,
        (-np.sin(yaw0) * delta[..., 0] + np.cos(yaw0) * delta[..., 1]) / 50.0,
        delta[..., 2] / 50.0,
        np.arctan2(np.sin(relative_yaw), np.cos(relative_yaw)) / np.pi,
        (pitch - pitch0) / .5,
        speed / 15.0,
        valid.astype(float),
    ), axis=-1)
    return windows.astype(np.float32)


def generate(output_dir, per_family=32, seed=20268, spacing_m=1.0,
             start_step_m=5.0, preview_step_m=4.0, preview_points=12):
    """Write path CSVs, split manifests and encoder windows to a new directory."""
    if per_family < 3:
        raise ValueError("per_family must be at least 3 for train/val/test splits")
    if min(spacing_m, start_step_m, preview_step_m) <= 0 or preview_points < 2:
        raise ValueError("spacing and steps must be positive; preview_points >= 2")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifests = []
    windows_by_split = {split: [] for split in ("train", "val", "test")}
    holdout = max(1, round(.1 * per_family))
    if 2 * holdout >= per_family:
        holdout = 1

    for family in FAMILIES:
        assignments = np.array(["train"] * (per_family - 2 * holdout) +
                               ["val"] * holdout + ["test"] * holdout)
        rng.shuffle(assignments)
        for index, split in enumerate(assignments):
            path_seed = int(rng.integers(0, 2**32, dtype=np.uint32))
            path_rng = np.random.default_rng(path_seed)
            columns, parameters = make_path(family, path_rng, spacing_m)
            path_id = f"{family}_{index:04d}"
            relative_file = Path(split) / f"{path_id}.csv"
            path_file = output_dir / relative_file
            path_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path_file, columns, delimiter=",", fmt="%.9f",
                       header=PATH_COLUMNS, comments="")
            path = LoopPath(path_file)
            windows_by_split[split].append(extract_windows(
                path, start_step_m, preview_step_m, preview_points))
            manifests.append({
                "id": path_id, "split": str(split), "family": family,
                "file": relative_file.as_posix(), "seed": path_seed,
                "path_length_m": path.length,
                "window_count": len(windows_by_split[split][-1]),
                "min_speed_mps": float(np.min(columns[:, 3])),
                "max_speed_mps": float(np.max(columns[:, 3])),
                "min_depth_m": float(np.min(columns[:, 2])),
                "max_depth_m": float(np.max(columns[:, 2])),
                "parameters": parameters,
            })

    for split, groups in windows_by_split.items():
        np.save(output_dir / f"{split}_windows.npy", np.concatenate(groups))
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for entry in manifests:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    metadata = {
        "seed": seed, "per_family": per_family, "families": FAMILIES,
        "spacing_m": spacing_m, "window_start_step_m": start_step_m,
        "preview_step_m": preview_step_m, "preview_points": preview_points,
        "window_features": WINDOW_FEATURES,
        "window_shape": [preview_points, len(WINDOW_FEATURES)],
        "split_counts": {split: {"paths": sum(row["split"] == split for row in manifests),
                                "windows": sum(len(group) for group in groups)}
                         for split, groups in windows_by_split.items()},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("path_corpus"))
    parser.add_argument("--per-family", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20268)
    parser.add_argument("--spacing-m", type=float, default=1.0)
    parser.add_argument("--window-start-step-m", type=float, default=5.0)
    parser.add_argument("--preview-step-m", type=float, default=4.0)
    parser.add_argument("--preview-points", type=int, default=12)
    args = parser.parse_args()
    metadata = generate(args.output_dir, args.per_family, args.seed,
                        args.spacing_m, args.window_start_step_m,
                        args.preview_step_m, args.preview_points)
    print(f"Corpus written to {args.output_dir}")
    for split, counts in metadata["split_counts"].items():
        print(f"  {split}: {counts['paths']} paths, {counts['windows']} windows")


if __name__ == "__main__":
    main()
