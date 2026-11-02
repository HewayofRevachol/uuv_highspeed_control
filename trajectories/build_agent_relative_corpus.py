"""Build body-relative path windows from an existing trajectory corpus.

The same ``agent_relative_window`` function can be called in the RL environment
with its current state and path progress. Synthetic offsets and attitude errors
make the offline encoder corpus cover states away from the reference path.

Usage:
    python3 trajectories/build_agent_relative_corpus.py trajectories/path_corpus
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.model import LoopPath, rotation_ned
from trajectories.add_encoder_targets import AUXILIARY_FEATURES, geometry_targets


INPUT_FEATURES = (
    "body_forward_50m", "body_starboard_50m", "body_down_50m",
    "path_yaw_relative_to_agent_pi", "path_pitch_relative_to_agent_0p5rad",
    "valid",
)
RECONSTRUCTION_FEATURES = INPUT_FEATURES[:-1]
ERROR_LEVELS = (
    # along, cross, depth in metres; yaw and pitch in radians.
    (.0, .0, .0, .0, .0),
    (2.0, 2.0, 1.5, np.deg2rad(5.0), np.deg2rad(3.0)),
    (5.0, 6.0, 4.0, np.deg2rad(12.0), np.deg2rad(8.0)),
    (10.0, 15.0, 10.0, np.deg2rad(25.0), np.deg2rad(15.0)),
)


def agent_relative_window(path, state, progress_m, offsets_m):
    """Return K×6 body-frame points, relative tangent angles and valid flags.

    ``state`` follows simulator.model's 10-state order. The body x/y/z axes
    are forward/starboard/down. ``progress_m`` is the path progress used by
    the controller, not a recomputed nearest-point projection.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (10,):
        raise ValueError("state must have shape (10,)")
    offsets_m = np.asarray(offsets_m, dtype=float)
    requested = progress_m + offsets_m
    valid = requested <= path.length
    sampled = np.clip(requested, 0.0, path.length)
    points = np.stack([np.interp(sampled, path.distances, axis)
                       for axis in path.points.T], axis=-1)
    body_relative = (points - state[:3]) @ rotation_ned(state[3], state[4])
    path_yaw = np.interp(sampled, path.distances, path.yaw)
    path_pitch = np.interp(sampled, path.distances, path.pitch)
    yaw_error = np.arctan2(np.sin(path_yaw - state[4]),
                           np.cos(path_yaw - state[4])) / np.pi
    pitch_error = (path_pitch - state[3]) / 0.5
    return np.column_stack((body_relative / 50.0,
                            yaw_error, pitch_error,
                            valid.astype(float))).astype(np.float32)


def _sample_state(path, progress_m, variant, rng):
    reference = path.position_at(progress_m)
    yaw = float(np.interp(progress_m, path.distances, path.yaw))
    pitch = path.pitch_at(progress_m)
    if variant == 0:
        errors = np.zeros(5)
    else:
        level = rng.choice((1, 2, 3), p=(.60, .30, .10))
        errors = rng.normal(size=5) * ERROR_LEVELS[level]
        errors = np.clip(errors, -2.5 * np.asarray(ERROR_LEVELS[level]),
                         2.5 * np.asarray(ERROR_LEVELS[level]))
    along, cross, depth, yaw_error, pitch_error = errors
    north = along * np.cos(yaw) - cross * np.sin(yaw)
    east = along * np.sin(yaw) + cross * np.cos(yaw)
    position = reference + np.array((north, east, depth))
    return np.r_[position, pitch + pitch_error, yaw + yaw_error,
                 np.zeros(5)]


def _save_array(path, values):
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def build(corpus_dir, variants_per_anchor=2, seed=20268):
    if variants_per_anchor < 1:
        raise ValueError("variants_per_anchor must be positive")
    corpus_dir = Path(corpus_dir)
    metadata_file = corpus_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if "encoder_dataset" not in metadata:
        raise ValueError("run add_encoder_targets.py before building agent windows")
    rows = [json.loads(line) for line in
            (corpus_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    start_step = float(metadata["window_start_step_m"])
    preview_step = float(metadata["preview_step_m"])
    preview_points = int(metadata["preview_points"])
    offsets = preview_step * np.arange(preview_points)
    counts = {}

    for split in ("train", "val", "test"):
        inputs, targets, speed_profiles = [], [], []
        for row in (item for item in rows if item["split"] == split):
            path = LoopPath(corpus_dir / row["file"])
            starts = np.arange(0.0, path.length, start_step)
            features = geometry_targets(path, start_step, preview_step,
                                        preview_points)
            if len(starts) != row["window_count"] or len(features) != len(starts):
                raise ValueError(f"window count differs for {row['file']}")
            rng = np.random.default_rng(np.random.SeedSequence((seed, row["seed"])))
            for index, progress in enumerate(starts):
                sampled = np.minimum(progress + offsets, path.length)
                speed = np.interp(sampled, path.distances, path.speed)
                for variant in range(variants_per_anchor):
                    state = _sample_state(path, progress, variant, rng)
                    window = agent_relative_window(path, state, progress, offsets)
                    inputs.append(window)
                    targets.append(np.concatenate((window[:, :-1], features[index]),
                                                  axis=-1))
                    speed_profiles.append((speed / 15.0)[:, None])
        input_array = np.stack(inputs).astype(np.float32)
        target_array = np.stack(targets).astype(np.float32)
        speed_array = np.stack(speed_profiles).astype(np.float32)
        _save_array(corpus_dir / f"{split}_agent_encoder_inputs.npy", input_array)
        _save_array(corpus_dir / f"{split}_agent_encoder_targets.npy", target_array)
        _save_array(corpus_dir / f"{split}_agent_speed_profiles.npy", speed_array)
        counts[split] = len(input_array)

    metadata["agent_relative_encoder_dataset"] = {
        "input_file": "{split}_agent_encoder_inputs.npy",
        "target_file": "{split}_agent_encoder_targets.npy",
        "speed_file": "{split}_agent_speed_profiles.npy",
        "input_features": INPUT_FEATURES,
        "reconstruction_features": RECONSTRUCTION_FEATURES,
        "target_features": RECONSTRUCTION_FEATURES + AUXILIARY_FEATURES,
        "input_shape_per_window": [preview_points, len(INPUT_FEATURES)],
        "target_shape_per_window": [preview_points,
                                    len(RECONSTRUCTION_FEATURES) + len(AUXILIARY_FEATURES)],
        "variants_per_anchor": variants_per_anchor,
        "synthetic_state_seed": seed,
        "axis_convention": "body forward/starboard/down",
        "position_scale_m": 50.0,
        "yaw_scale_rad": float(np.pi),
        "pitch_scale_rad": 0.5,
        "counts": counts,
        "note": "A clean path-aligned pose and perturbed poses are included per anchor; speed is separate.",
    }
    temporary = metadata_file.with_name("metadata.tmp.json")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(metadata_file)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("--variants-per-anchor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20268)
    args = parser.parse_args()
    counts = build(args.corpus_dir, args.variants_per_anchor, args.seed)
    for split, count in counts.items():
        print(f"{split}: {count} body-relative path windows")


if __name__ == "__main__":
    main()
