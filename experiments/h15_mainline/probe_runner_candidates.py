"""Capture candidate costs from the Gate1 runner for diagnostic replay."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from experiments.h15_mainline import run_offline_governor_smoke as runner
from experiments.h15_mainline.candidate_filter import _scenarios
from experiments.h15_mainline.probe_coupled_rejections import envelope_radii
from types import SimpleNamespace
from simulator.model import LoopPath


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", choices=("simple", "coupled", "mismatch"), required=True)
    parser.add_argument("--method", choices=("B0_shared", "B2_gov"), required=True)
    parser.add_argument("--max-periods", type=int, required=True)
    parser.add_argument("--envelope", type=Path)
    parser.add_argument("--qp-slew", action="store_true")
    parser.add_argument("--qp-pitch", action="store_true")
    parser.add_argument("--qp-speed", action="store_true")
    parser.add_argument("--qp-iterations", type=int, default=1)
    parser.add_argument("--qp-pitch-buffer-deg", type=float, default=0.)
    parser.add_argument("--qp-speed-buffer-mps", type=float, default=0.)
    parser.add_argument("--capture-period", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    captures = []
    screen = runner.screen_candidate

    def capture(env, path, candidate, estimate, config, bounds):
        period = len(captures) // len(config["candidate_grid_mps"])
        if args.qp_pitch:
            env.mpc.set_screen_progress(float(env.progress[0]))
        before = {"reference_value": env._reference_value,
                  "reference_step": env._reference_step,
                  "plan_first": env.mpc.plan[0, 0].tolist(),
                  "previous": env.mpc.previous[0].tolist(),
                  "current_estimate": env.current_estimate[0].tolist(),
                  "progress_m": float(env.progress[0])}
        result = screen(env, path, candidate, estimate, config, bounds)
        item = {"period": period, "candidate_mps": candidate,
                "state": before, "cost": result.get("cost"),
                "accepted": result["physical_accepted"],
                "first_rejection": result["first_rejection"]}
        if period == args.capture_period:
            item["scenarios"] = [{"name": scenario["scenario"],
                                  "first_rejection": scenario["first_rejection"],
                                  "first_time_s": scenario["first_time_s"],
                                  "maxima": scenario["maxima"],
                                  "min_margins": scenario["min_margins"]}
                                 for scenario in result["scenarios"]]
        captures.append(item)
        return result

    runner.screen_candidate = capture
    if args.qp_speed and not args.qp_pitch:
        raise ValueError("speed diagnostic requires five-scenario pitch branch")
    if args.qp_slew or args.qp_pitch or args.qp_iterations != 1:
        create_env = runner.create_env
        config = json.loads(args.config.read_text())
        bounds = json.loads(args.checkpoint.read_text())["parameters"]
        if args.qp_pitch and args.envelope is None:
            raise ValueError("pitch diagnostic requires envelope residuals")
        pitch_radii = envelope_radii(json.loads(args.envelope.read_text())) if args.qp_pitch else None
        estimate = SimpleNamespace(fin_gain=bounds["e0_gain"],
                                   fin_tau_scale=bounds["e0_tau_scale"],
                                   gain_radius=max(bounds["e0_gain"]-bounds["gain_bounds"][0],
                                                   bounds["gain_bounds"][1]-bounds["e0_gain"]),
                                   tau_radius=max(bounds["e0_tau_scale"]-bounds["tau_scale_bounds"][0],
                                                  bounds["tau_scale_bounds"][1]-bounds["e0_tau_scale"]))
        scenarios = [(gain, tau) for _, gain, tau in _scenarios(estimate, bounds)]
        pitch_limits = []
        speed_limits = []
        if args.qp_pitch:
            for period in range(1, config["horizon_periods"] + 1):
                horizon = next(h for h in (1, 5, 25, 35) if period <= h)
                radius = pitch_radii[str(horizon)]["pitch_deg"]
                terminal = config["terminal_extra_margin"]["pitch_deg"] if period == config["horizon_periods"] else 0.
                pitch_limits.append(config["strict_limits"]["pitch_deg"]-radius-terminal
                                    -args.qp_pitch_buffer_deg)
                speed_radius = pitch_radii[str(horizon)]["tangent_water_speed_mps"]
                speed_terminal = (config["terminal_extra_margin"]["tangent_water_speed_error_mps"]
                                  if period == config["horizon_periods"] else 0.)
                speed_limits.append(config["strict_limits"]["tangent_water_speed_error_mps"]
                                    -speed_radius-speed_terminal-args.qp_speed_buffer_mps
                                    if args.qp_speed else 1e6)

        def create_slew_env(path_file, seed, mismatch):
            env = create_env(path_file, seed, mismatch)
            if args.qp_slew:
                env.mpc.set_command_slew_limits(config["command_max_delta_normalized_per_period"])
            if args.qp_pitch:
                env.mpc.set_screen_scenarios(scenarios, pitch_limits,
                                             speed_limits, LoopPath(path_file))
            env.mpc.set_sqp_iterations(args.qp_iterations)
            return env

        runner.create_env = create_slew_env
    if args.envelope is not None:
        load = runner.load_frozen_estimator
        radii = envelope_radii(json.loads(args.envelope.read_text()))

        def load_envelope(checkpoint_path):
            estimator = load(checkpoint_path)
            estimator.constraint_radii = radii
            return estimator

        runner.load_frozen_estimator = load_envelope
    sys.argv = ["run_offline_governor_smoke", "--freeze", str(args.freeze),
                "--checkpoint", str(args.checkpoint), "--config", str(args.config),
                "--case", args.case, "--method", args.method,
                "--max-periods", str(args.max_periods), "--out", str(args.out)]
    runner.main()
    provenance = {"schema": 1, "scope": "diagnostic wrapper, not formal Gate1",
                  "runner_sha256": sha(ROOT / "experiments/h15_mainline/run_offline_governor_smoke.py"),
                  "probe_sha256": sha(Path(__file__)),
                  "condensed_mpc_sha256": sha(ROOT / "aav_mpc/condensed_mpc.py"),
                  "qp_slew": args.qp_slew,
                  "qp_pitch": args.qp_pitch,
                  "qp_speed": args.qp_speed,
                  "qp_iterations": args.qp_iterations,
                  "qp_pitch_buffer_deg": args.qp_pitch_buffer_deg,
                  "qp_speed_buffer_mps": args.qp_speed_buffer_mps,
                  "capture_period": args.capture_period,
                  "envelope_sha256": None if args.envelope is None else sha(args.envelope),
                  "candidates": captures}
    (args.out / "candidate_probe.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps([item for item in captures if item["period"] == 12], indent=2))


if __name__ == "__main__":
    main()
