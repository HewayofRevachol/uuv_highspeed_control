"""Load a frozen train/dev E_selected checkpoint with hash checks."""

import hashlib
import json
from pathlib import Path

from experiments.h15_mainline.response_estimator import SlidingWindowResponseEstimator


ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_estimator(checkpoint_path):
    checkpoint = json.loads(Path(checkpoint_path).read_text())
    if checkpoint.get("method") != "E_selected_E1_bounded_window_wls":
        raise ValueError("unexpected estimator checkpoint")
    normalization_path = ROOT / checkpoint["normalization_path"]
    calibration_path = ROOT / checkpoint["calibration_path"]
    if _sha(normalization_path) != checkpoint["normalization_sha256"]:
        raise ValueError("training normalization hash mismatch")
    if _sha(calibration_path) != checkpoint["calibration_sha256"]:
        raise ValueError("dev calibration hash mismatch")
    source = ROOT / "experiments/h15_mainline/response_estimator.py"
    if _sha(source) != checkpoint["source_sha256"]:
        raise ValueError("estimator source hash mismatch")
    normalization = json.loads(normalization_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    return SlidingWindowResponseEstimator(normalization, checkpoint["parameters"],
                                          calibration)
