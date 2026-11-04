"""Deployable H15 response-history features and train-only normalization."""

import json
import math


FEATURE_NAMES = (
    "lateral_error_m", "depth_error_m", "speed_error_mps", "pitch_deg",
    "actual_thrust_fraction", "actual_elevator_rad", "actual_rudder_rad",
    "issued_thrust_fraction", "issued_elevator_rad", "issued_rudder_rad",
    "previous_queue_thrust_fraction", "previous_queue_elevator_rad", "previous_queue_rudder_rad",
)
THRUST_SCALE_N = 1_500_000.0


def from_period_row(row):
    """Map a diagnostic row to observations available at deployment time."""
    actual = (float(row["thrust_actual_n"]),
              math.radians(float(row["elevator_actual_deg"])),
              math.radians(float(row["rudder_actual_deg"])))
    issued = json.loads(row["command_json"])
    queued = json.loads(row["queue_before_json"])[0]
    values = (float(row["lateral_error_m"]), float(row["depth_error_m"]),
              float(row["speed_error_mps"]), float(row["pitch_deg"]),
              actual[0] / THRUST_SCALE_N, actual[1], actual[2],
              issued[0] / THRUST_SCALE_N, issued[1], issued[2],
              queued[0] / THRUST_SCALE_N, queued[1], queued[2])
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite or malformed deployable observation")
    return values
