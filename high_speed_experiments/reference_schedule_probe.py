"""Matched zero-residual MPC probe for speed-aware path references.

This is a mechanism screen, not an RL result. The path target and success
criteria remain fixed while only the speed used by the MPC reference changes.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DDAT_train import ExperimentDDATPathEnv
from simulator.batch_path_env import PathBank


DEFAULT_PATHS = sorted((ROOT / "DDAT_trajectories/val").glob("*.csv"))


class ScheduledReferenceEnv(ExperimentDDATPathEnv):
    def __init__(self, *args, speed_floor=None, speed_cap=None,
                 preview_scale=1., curvature_accel_cap=None, **kwargs):
        self.speed_floor = speed_floor
        self.speed_cap = speed_cap
        self.preview_scale = preview_scale
        self.curvature_accel_cap = curvature_accel_cap
        super().__init__(*args, **kwargs)

    def _reference(self, time_offset=0.):
        offsets = torch.arange(1, self.mpc.horizon + 1, device=self.device,
                               dtype=self.dtype) * self.mpc.dt
        curve_cap = None
        if self.curvature_accel_cap is not None:
            ahead = self.progress[:, None] + self.model.tensor([0., 10., 20., 30.])
            _, _, _, yaw_ahead, _ = self.paths.sample(self.path_ids, ahead)
            delta = torch.atan2(torch.sin(yaw_ahead[:, 1:] - yaw_ahead[:, :-1]),
                                torch.cos(yaw_ahead[:, 1:] - yaw_ahead[:, :-1]))
            curvature = (delta.abs() / 10.).amax(-1).clamp_min(1e-6)
            curve_cap = torch.sqrt(self.curvature_accel_cap / curvature).clamp_max(18.)
        _, _, _, _, speed = self.paths.sample(self.path_ids, self.progress[:, None])
        speed = self._scheduled_speed(speed)
        if curve_cap is not None:
            speed = torch.minimum(speed, curve_cap[:, None])
        distances = self.progress[:, None] + speed.clamp_min(.1) * (
            self.preview_scale * offsets[None] + time_offset)
        position, tangent, pitch, yaw, target_speed = self.paths.sample(
            self.path_ids, distances)
        target_speed = self._scheduled_speed(target_speed)
        if curve_cap is not None:
            target_speed = torch.minimum(target_speed, curve_cap[:, None])
        velocity = target_speed[..., None] * tangent
        reference = torch.cat((position, pitch[..., None], yaw[..., None], velocity), -1)
        reference[..., :3] -= offsets[None, :, None] * self.current_estimate[:, None]
        reference[..., 5:8] -= self.current_estimate[:, None]
        return reference

    def _scheduled_speed(self, speed):
        if self.speed_floor is not None:
            speed = speed.clamp_min(self.speed_floor)
        if self.speed_cap is not None:
            speed = speed.clamp_max(self.speed_cap)
        return speed


SCHEDULES = {
    "fixed": {},
    "floor8": {"speed_floor": 8.},
    "floor9": {"speed_floor": 9.},
    "floor10": {"speed_floor": 10.},
    "floor12": {"speed_floor": 12.},
    "cap17": {"speed_cap": 17.},
    "cap15": {"speed_cap": 15.},
    "band9_15": {"speed_floor": 9., "speed_cap": 15.},
    "preview15": {"preview_scale": 1.5},
    "curv6": {"curvature_accel_cap": 6.},
}


@torch.no_grad()
def evaluate(file, speed, schedule, seed, backend, horizon, current_max,
             randomize_plant):
    bank = PathBank([file], "cpu")
    bank.speed[:] = speed
    bank.first_state[:, 5] = speed
    bank.durations[:] = bank.lengths / speed
    env = ScheduledReferenceEnv(
        bank, batch_size=1, device="cpu", horizon=horizon, backend=backend,
        seed=seed, randomize_plant=randomize_plant, current_max_mps=current_max,
        admm_iterations=400, admm_rho=10., attitude_pitch_weight=100.,
        attitude_yaw_weight=30., fsp_enabled=True, preview_mode="multiscale",
        yaw_reward="course", **SCHEDULES[schedule])
    env.reset(torch.zeros(1, device=env.device, dtype=torch.long))
    errors = []
    speeds = []
    pitches = []
    fallbacks = 0
    saturation = 0
    total = 0
    start = time.perf_counter()
    while True:
        _, _, done, info = env.step(torch.zeros(1, 3, device=env.device))
        errors.append(info["position_error_m"][0].item())
        speeds.append(env.state[0, 5:8].norm().item())
        pitches.append(abs(env.state[0, 3].item()))
        fallbacks += int(not info["mpc_ok"][0].item())
        command = info["command"][0]
        upper = env.model.upper * env.model.control_scale
        lower = env.model.lower * env.model.control_scale
        saturation += int(bool(((command >= upper - 1e-6) |
                                (command <= lower + 1e-6)).any()))
        total += 1
        if done[0].item():
            break
    return {
        "path": file.stem, "target_speed_mps": speed, "schedule": schedule,
        "seed": seed, "backend": backend, "horizon": horizon,
        "current_max_mps": current_max,
        "randomize_plant": randomize_plant,
        "success": bool(info["success"][0].item()),
        "failed": bool(info["failed"][0].item()),
        "timeout": bool(info["timeout"][0].item()),
        "steps": total, "progress_fraction": info["progress_fraction"][0].item(),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "position_max_m": float(np.max(errors)),
        "speed_rmse_mps": float(np.sqrt(np.mean(np.square(np.asarray(speeds) - speed)))),
        "max_body_speed_mps": float(np.max(speeds)),
        "max_abs_pitch_deg": float(np.rad2deg(np.max(pitches))),
        "fallback_fraction": fallbacks / total,
        "saturation_fraction": saturation / total,
        "wall_s": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--speeds", nargs="+", type=float, default=[6., 18.])
    parser.add_argument("--schedules", nargs="+", choices=tuple(SCHEDULES),
                        default=["fixed", "floor9", "cap15", "band9_15", "preview15"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101026])
    parser.add_argument("--backend", choices=("admm", "osqp"), default="osqp")
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--current-max", type=float, default=0.)
    parser.add_argument("--randomize-plant", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon <= 0 or any(speed <= 0 for speed in args.speeds):
        parser.error("horizon and speeds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for file in args.paths:
            for speed in args.speeds:
                for seed in args.seeds:
                    for schedule in args.schedules:
                        row = evaluate(file, speed, schedule, seed, args.backend,
                                       args.horizon, args.current_max,
                                       args.randomize_plant)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        print(f"{row['path']} v={speed:g} seed={seed} {schedule}: "
                              f"success={row['success']} failed={row['failed']} "
                              f"progress={row['progress_fraction']:.3f} "
                              f"RMSE={row['position_rmse_m']:.2f}m "
                              f"fallback={row['fallback_fraction']:.2f} "
                              f"wall={row['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
