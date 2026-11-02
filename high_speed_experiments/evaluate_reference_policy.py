"""Matched evaluation of a trained reference-speed actor and zero-action MPC."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from high_speed_experiments.train_reference_policy import (
    SpeedActorCritic, make_env, speed_bank)


DEFAULT_PATHS = sorted((ROOT / "DDAT_trajectories/val").glob("*.csv"))


@torch.no_grad()
def evaluate(file, speed, seed, mode, actor, backend, config):
    bank = speed_bank([file], "cpu", speeds=(speed,))
    env = make_env(bank, 1, backend, seed, randomize_plant=True,
                   current_max=1., admm_iterations=200,
                   base_reference=config.get("base_reference", "task"),
                   residual_scale=config.get("residual_scale", 2.),
                   residual_penalty=config.get("residual_penalty", 0.))
    observation = env.reset(torch.zeros(1, dtype=torch.long))
    errors = []
    speed_errors = []
    reference_speeds = []
    saturation = 0
    fallback = 0
    while True:
        action = (actor.deterministic(observation) if mode == "policy" else
                  torch.zeros(1, 1))
        observation, _, done, info = env.step(action)
        errors.append(info["position_error_m"][0].item())
        speed_errors.append(env.state[0, 5:8].norm().item() - speed)
        reference_speeds.append(info["reference_speed_mps"][0].item())
        command = info["command"][0]
        upper = env.model.upper * env.model.control_scale
        lower = env.model.lower * env.model.control_scale
        saturation += int(bool(((command >= upper - 1e-6) |
                                (command <= lower + 1e-6)).any()))
        fallback += int(not info["mpc_ok"][0].item())
        if done[0].item():
            break
    steps = len(errors)
    return {
        "path": file.stem, "target_speed_mps": speed, "seed": seed,
        "mode": mode, "backend": backend,
        "success": bool(info["success"][0].item()),
        "failed": bool(info["failed"][0].item()),
        "timeout": bool(info["timeout"][0].item()),
        "steps": steps, "progress_fraction": info["progress_fraction"][0].item(),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "position_max_m": float(np.max(errors)),
        "speed_rmse_mps": float(np.sqrt(np.mean(np.square(speed_errors)))),
        "reference_speed_mean_mps": float(np.mean(reference_speeds)),
        "reference_speed_min_mps": float(np.min(reference_speeds)),
        "saturation_fraction": saturation / steps,
        "fallback_fraction": fallback / steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paths", nargs="+", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--speeds", nargs="+", type=float, default=[18.])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101026, 101027])
    parser.add_argument("--modes", nargs="+", choices=("zero", "policy"),
                        default=["zero", "policy"])
    parser.add_argument("--backend", choices=("admm", "osqp"), default="osqp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    actor = SpeedActorCritic(checkpoint["observation_dim"])
    actor.load_state_dict(checkpoint["agent"])
    actor.eval()
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for file in args.paths:
            for speed in args.speeds:
                for seed in args.seeds:
                    for mode in args.modes:
                        row = evaluate(file, speed, seed, mode, actor, args.backend,
                                       checkpoint["config"])
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        print(f"{file.stem} v={speed:g} seed={seed} {mode}: "
                              f"success={row['success']} "
                              f"RMSE={row['position_rmse_m']:.2f}m "
                              f"speedRMSE={row['speed_rmse_mps']:.2f}m/s "
                              f"ref={row['reference_speed_mean_mps']:.2f}m/s", flush=True)


if __name__ == "__main__":
    main()
