"""Add geometry targets to an existing path corpus for encoder pretraining.

Each train/val/test path CSV gains four geometric feature columns. The encoder
input contains geometry and the validity mask. Its target contains the same
geometry plus the four path-derived features. The prescribed speed profile is
saved separately because it cannot be inferred from path geometry alone.

Usage:
    python3 trajectories/add_encoder_targets.py trajectories/path_corpus
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.model import LoopPath
from trajectories.generate_path_corpus import PATH_COLUMNS, extract_windows


INPUT_INDICES = (0, 1, 2, 3, 4, 6)
GEOMETRY_FEATURES = (
    "forward_50m", "lateral_50m", "depth_50m", "yaw_pi",
    "pitch_0p5rad", "valid",
)
AUXILIARY_FEATURES = (
    "signed_heading_change_per_m_div_0p02",
    "pitch_change_per_m_div_0p02",
    "turn_entry_score",
    "depth_reversal_score",
)
PATH_FEATURE_COLUMNS = (
    "signed_heading_change_per_m", "pitch_change_per_m",
    "turn_entry_score", "depth_reversal_score",
)
CURVATURE_SCALE_PER_M = 0.02
TURN_THRESHOLD_PER_M = 0.003
PITCH_THRESHOLD_RAD = 0.02
EVENT_WIDTH_M = 4.0
SMOOTH_WIDTH_M = 2.0


def _smoothed_gradient(values, distance):
    """Differentiate with respect to 3D path arc length, then smooth in metres."""
    step = float(np.median(np.diff(distance)))
    derivative = np.gradient(values, distance)
    derivative = gaussian_filter1d(
        derivative, sigma=max(0.5, SMOOTH_WIDTH_M / step), mode="nearest")
    derivative[np.abs(derivative) < 1e-8] = 0.0
    return derivative


def _entry_positions(signal, distance, threshold, reversal_only=False):
    """Find significant signed runs; optionally keep only direction reversals."""
    active = np.where(signal > threshold, 1,
                      np.where(signal < -threshold, -1, 0))
    starts = np.flatnonzero((active != 0) &
                            (active != np.r_[0, active[:-1]]))
    if not reversal_only:
        return distance[starts]
    reversals = []
    previous_direction = None
    for index in starts:
        direction = int(active[index])
        if previous_direction is not None and direction != previous_direction:
            reversals.append(distance[index])
        previous_direction = direction
    return np.asarray(reversals, dtype=float)


def _event_scores(distance, events):
    """Smoothly mark event positions on the original path sample grid."""
    scores = np.zeros_like(distance, dtype=float)
    for event in events:
        bump = np.exp(-0.5 * ((distance - event) / EVENT_WIDTH_M)**2)
        scores = np.maximum(scores, bump)
    return scores


def path_geometry_profile(path):
    """Return four physical-unit features at each original path CSV row."""
    distance = path.distances
    heading_change = _smoothed_gradient(path.yaw, distance)
    pitch_change = _smoothed_gradient(path.pitch, distance)
    turn_entries = _entry_positions(
        heading_change, distance, TURN_THRESHOLD_PER_M)
    depth_reversals = _entry_positions(
        path.pitch, distance, PITCH_THRESHOLD_RAD, reversal_only=True)
    return np.stack((heading_change, pitch_change,
                     _event_scores(distance, turn_entries),
                     _event_scores(distance, depth_reversals)), axis=-1)


def geometry_targets(path, start_step_m, preview_step_m, preview_points,
                     profile=None):
    """Interpolate CSV geometry features into normalized path windows."""
    distance = path.distances
    if profile is None:
        profile = path_geometry_profile(path)

    starts = np.arange(0.0, path.length, start_step_m)
    requested = starts[:, None] + preview_step_m * np.arange(preview_points)
    valid = requested <= path.length
    sampled = np.minimum(requested, path.length)
    features = np.stack([np.interp(sampled, distance, profile[:, index])
                         for index in range(len(PATH_FEATURE_COLUMNS))], axis=-1)
    features[..., :2] /= CURVATURE_SCALE_PER_M
    features *= valid[..., None]
    return features.astype(np.float32)


def _write_enriched_path(path_file, profile):
    """Replace a path CSV with its original four columns plus feature columns."""
    with path_file.open("r", newline="", encoding="utf-8") as stream:
        header = next(csv.reader(stream))
    base_columns = tuple(PATH_COLUMNS.split(","))
    if tuple(header[:4]) != base_columns:
        raise ValueError(f"unexpected trajectory columns in {path_file}")
    original = np.loadtxt(path_file, delimiter=",", skiprows=1, ndmin=2)
    if len(original) != len(profile):
        raise ValueError(f"trajectory and feature row counts differ: {path_file}")
    enriched = np.column_stack((original[:, :4], profile))
    temporary = path_file.with_name(path_file.stem + ".tmp.csv")
    np.savetxt(temporary, enriched, delimiter=",", fmt="%.9f",
               header=",".join(base_columns + PATH_FEATURE_COLUMNS), comments="")
    temporary.replace(path_file)


def _save_array(path, array):
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, array)
    temporary.replace(path)


def augment(corpus_dir):
    """Enrich path CSVs and write aligned input, target and speed arrays."""
    corpus_dir = Path(corpus_dir)
    metadata_file = corpus_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (corpus_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    start_step = float(metadata["window_start_step_m"])
    preview_step = float(metadata["preview_step_m"])
    preview_points = int(metadata["preview_points"])
    if tuple(metadata["window_features"]) != (
        "forward_50m", "lateral_50m", "depth_50m", "yaw_pi",
        "pitch_0p5rad", "target_speed_15mps", "valid"
    ):
        raise ValueError("the corpus window schema is not supported")

    counts = {}
    for split in ("train", "val", "test"):
        raw = np.load(corpus_dir / f"{split}_windows.npy", mmap_mode="r")
        paths = [row for row in rows if row["split"] == split]
        if len(raw) != sum(row["window_count"] for row in paths):
            raise ValueError(f"{split}: manifest and window count differ")
        inputs, targets, speeds = [], [], []
        offset = 0
        for row in paths:
            path = LoopPath(corpus_dir / row["file"])
            expected = extract_windows(path, start_step, preview_step,
                                       preview_points)
            count = len(expected)
            source = np.asarray(raw[offset:offset + count])
            if source.shape != expected.shape or not np.allclose(source, expected):
                raise ValueError(f"raw windows do not match {row['file']}")
            geometry = source[..., INPUT_INDICES]
            profile = path_geometry_profile(path)
            features = geometry_targets(path, start_step, preview_step,
                                        preview_points, profile)
            inputs.append(geometry)
            targets.append(np.concatenate((geometry, features), axis=-1))
            speeds.append(source[..., 5:6])
            _write_enriched_path(corpus_dir / row["file"], profile)
            offset += count

        _save_array(corpus_dir / f"{split}_encoder_inputs.npy",
                    np.concatenate(inputs).astype(np.float32))
        _save_array(corpus_dir / f"{split}_encoder_targets.npy",
                    np.concatenate(targets).astype(np.float32))
        _save_array(corpus_dir / f"{split}_speed_profiles.npy",
                    np.concatenate(speeds).astype(np.float32))
        counts[split] = len(raw)

    metadata["encoder_dataset"] = {
        "input_file": "{split}_encoder_inputs.npy",
        "target_file": "{split}_encoder_targets.npy",
        "speed_file": "{split}_speed_profiles.npy",
        "input_features": GEOMETRY_FEATURES,
        "target_features": GEOMETRY_FEATURES + AUXILIARY_FEATURES,
        "auxiliary_features": AUXILIARY_FEATURES,
        "path_csv_feature_columns": PATH_FEATURE_COLUMNS,
        "path_csv_feature_units": ("rad/m", "rad/m", "score_0_to_1", "score_0_to_1"),
        "input_shape_per_window": [preview_points, len(GEOMETRY_FEATURES)],
        "target_shape_per_window": [preview_points,
                                    len(GEOMETRY_FEATURES) + len(AUXILIARY_FEATURES)],
        "curvature_scale_per_m": CURVATURE_SCALE_PER_M,
        "turn_threshold_per_m": TURN_THRESHOLD_PER_M,
        "pitch_threshold_rad": PITCH_THRESHOLD_RAD,
        "event_width_m": EVENT_WIDTH_M,
        "smooth_width_m": SMOOTH_WIDTH_M,
        "note": "Mask target losses using the valid input channel. Speed is prescribed separately.",
    }
    temporary = metadata_file.with_name("metadata.tmp.json")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(metadata_file)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path,
                        help="directory written by generate_path_corpus.py")
    args = parser.parse_args()
    counts = augment(args.corpus_dir)
    for split, count in counts.items():
        print(f"{split}: {count} windows and path CSVs enriched with geometry features")


if __name__ == "__main__":
    main()
