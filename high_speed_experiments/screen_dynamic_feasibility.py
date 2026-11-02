"""Conservative steady-segment screen for fixed-speed UUV path references.

The screen rejects a *speed schedule* only when a sustained constant-speed,
curvature and depth-slope segment requires more steady thrust or fin angle than
the plant permits. Passing the screen is not a proof of transient feasibility.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulator.model import (D_COEFF, MAX_RUDDER, MISMATCHED_PLANT,
                             NOMINAL_PLANT, LoopPath, derivative)


MAX_THRUST_N = 1_500_000.0
SAMPLE_SPACING_M = 1.0
MIN_SEGMENT_M = 20.0
LIMIT_MARGIN = 0.01


def steady_planar_trim(speed, curvature, plant):
    """Return a constant-speed, constant-curvature horizontal equilibrium."""
    if abs(curvature) < 1e-8:
        return {"thrust_n": float(plant.damping_scale * D_COEFF[0, 0]
                                  * speed**2 / plant.thrust_gain),
                "rudder_rad": 0.0, "surge_mps": float(speed),
                "sway_mps": 0.0, "yaw_rate_radps": 0.0,
                "converged": True}

    yaw_rate = speed * curvature

    def residual(unknown):
        surge, sway, rudder, thrust = unknown
        state = np.zeros(10)
        state[5], state[6], state[9] = surge, sway, yaw_rate
        change = derivative(state, [thrust, 0.0, rudder], plant)
        return np.r_[change[[5, 6, 9]], np.hypot(surge, sway) - speed]

    straight_thrust = plant.damping_scale * D_COEFF[0, 0] * speed**2 / plant.thrust_gain
    initial = [speed, 0.0, np.sign(curvature) * np.deg2rad(10), straight_thrust]
    solution = root(residual, initial)
    converged = (solution.success and solution.x[0] > 0 and
                 np.linalg.norm(residual(solution.x)) < 1e-6)
    return {"thrust_n": float(solution.x[3]),
            "rudder_rad": float(solution.x[2]),
            "surge_mps": float(solution.x[0]),
            "sway_mps": float(solution.x[1]),
            "yaw_rate_radps": float(yaw_rate),
            "converged": bool(converged)}


def steady_spatial_trim(speed, curvature, path_pitch, plant):
    """Steady helical trim at a 3-D arc speed and course curvature per arc metre."""
    if abs(path_pitch) < 1e-6:
        return steady_planar_trim(speed, curvature, plant)
    target_horizontal = speed * np.cos(path_pitch)
    target_depth_rate = -speed * np.sin(path_pitch)

    def residual(unknown):
        surge, sway, heave, pitch, yaw_rate, thrust_mn, elevator, rudder = unknown
        ct, st = np.cos(pitch), np.sin(pitch)
        horizontal_forward = ct * surge + st * heave
        horizontal_speed = np.hypot(horizontal_forward, sway)
        depth_rate = -st * surge + ct * heave
        state = np.zeros(10)
        state[3] = pitch
        state[5:] = [surge, sway, heave, 0.0, yaw_rate]
        acceleration = derivative(state,
                                  [thrust_mn * 1e6, elevator, rudder], plant)
        return np.r_[horizontal_speed - target_horizontal,
                     depth_rate - target_depth_rate,
                     20 * (yaw_rate / ct - speed * curvature),
                     acceleration[5:8], 20 * acceleration[8:10]]

    straight_mn = (plant.damping_scale * D_COEFF[0, 0] * speed**2 /
                   plant.thrust_gain / 1e6)
    initial = [speed, 0.0, 0.0, path_pitch,
               speed * curvature * np.cos(path_pitch), straight_mn,
               0.0, np.sign(curvature) * np.deg2rad(10)]
    solution = root(residual, initial)
    surge, sway, heave, pitch, yaw_rate, thrust_mn, elevator, rudder = solution.x
    converged = (solution.success and surge > 0 and
                 abs(pitch) < np.deg2rad(70) and
                 np.linalg.norm(residual(solution.x)) < 1e-6)
    return {"thrust_n": float(thrust_mn * 1e6),
            "rudder_rad": float(rudder), "elevator_rad": float(elevator),
            "surge_mps": float(surge), "sway_mps": float(sway),
            "heave_mps": float(heave), "pitch_rad": float(pitch),
            "yaw_rate_radps": float(yaw_rate),
            "converged": bool(converged)}


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def screen_path(file, plant=MISMATCHED_PLANT):
    """Flag sustained steady demands beyond the model's command bounds."""
    path = LoopPath(file)
    distance = np.arange(0.0, path.length, SAMPLE_SPACING_M)
    if len(distance) < 3:
        raise ValueError(f"path is too short for screening: {file}")
    yaw = np.interp(distance, path.distances, path.yaw)
    pitch = np.interp(distance, path.distances, path.pitch)
    speed = np.interp(distance, path.distances, path.speed)
    curvature = np.gradient(yaw, distance)
    curvature_slope = np.gradient(curvature, distance)
    speed_slope = np.gradient(speed, distance)
    pitch_slope = np.gradient(pitch, distance)
    steady = ((np.abs(pitch_slope) < np.deg2rad(0.05)) &
              (np.abs(speed_slope) < 0.005) &
              (np.abs(curvature_slope) < 2e-5))

    excessive = np.zeros(len(distance), dtype=bool)
    unresolved = np.zeros(len(distance), dtype=bool)
    thrust = np.full(len(distance), np.nan)
    rudder = np.full(len(distance), np.nan)
    elevator = np.full(len(distance), np.nan)
    cache = {}
    for index in np.flatnonzero(steady):
        key = (round(float(speed[index]), 2), round(float(curvature[index]), 5),
               round(float(pitch[index]), 4))
        if key not in cache:
            cache[key] = steady_spatial_trim(*key, plant)
        trim = cache[key]
        if not trim["converged"]:
            unresolved[index] = True
            continue
        thrust[index] = trim["thrust_n"]
        rudder[index] = trim["rudder_rad"]
        elevator[index] = trim.get("elevator_rad", 0.0)
        excessive[index] = (thrust[index] > MAX_THRUST_N * (1 + LIMIT_MARGIN) or
                            abs(rudder[index]) > MAX_RUDDER * (1 + LIMIT_MARGIN) or
                            abs(elevator[index]) > MAX_RUDDER * (1 + LIMIT_MARGIN))

    segments = []
    for start, stop in _runs(excessive):
        if (stop - start) * SAMPLE_SPACING_M < MIN_SEGMENT_M:
            continue
        part = slice(start, stop)
        segments.append({
            "start_m": round(float(distance[start]), 1),
            "end_m": round(float(distance[stop - 1] + SAMPLE_SPACING_M), 1),
            "target_speed_mps": round(float(np.median(speed[part])), 2),
            "median_radius_m": (None if np.median(abs(curvature[part])) < 1e-8 else
                                round(float(1 / np.median(abs(curvature[part]))), 1)),
            "max_required_thrust_mn": round(float(np.max(thrust[part]) / 1e6), 3),
            "max_required_rudder_deg": round(float(np.rad2deg(np.max(abs(rudder[part])))), 2),
            "max_required_elevator_deg": round(float(np.rad2deg(np.max(abs(elevator[part])))), 2),
        })

    # Avoid the half-curvature endpoint artifact of LoopPath's tangent estimate.
    near_start_curvature = float(np.median(curvature[1:min(6, len(curvature))]))
    initial_required_r = float(np.cos(path.pitch[0]) * path.speed[0]
                               * near_start_curvature)
    return {
        "path": str(Path(file)),
        "status": "screened_out" if segments else "not_screened_out",
        "reason": "sustained_steady_command_limit" if segments else None,
        "length_m": round(float(path.length), 2),
        "max_target_speed_mps": round(float(speed.max()), 2),
        "near_start_required_yaw_rate_radps": round(initial_required_r, 4),
        "initial_yaw_rate_is_zero": True,
        "samples_outside_steady_screen": int((~steady).sum()),
        "unresolved_steady_samples": int(unresolved.sum()),
        "excess_segments": segments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--plant", choices=("mismatch", "nominal"), default="mismatch")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plant = MISMATCHED_PLANT if args.plant == "mismatch" else NOMINAL_PLANT
    files = [file for item in args.paths for file in
             (sorted(item.glob("*.csv")) if item.is_dir() else [item])]
    report = {"definition": "sustained steady spatial trim exceeds command bounds by >1%",
              "plant": args.plant, "max_thrust_n": MAX_THRUST_N,
              "max_rudder_deg": round(float(np.rad2deg(MAX_RUDDER)), 1),
              "minimum_excess_length_m": MIN_SEGMENT_M,
              "results": [screen_path(file, plant) for file in files]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
