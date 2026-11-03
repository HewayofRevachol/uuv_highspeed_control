"""Validate a DDAT checkpoint against its zero-action MPC baseline.

Run validation paths by default, or inspect the paths actually used to train:
    python3 validate_ddat.py
    python3 validate_ddat.py --split train --seeds 3

By default, use the newest checkpoint's validation path directory and its
recorded controller settings. Path screening and plant randomization follow
the checkpoint's training settings. Each pair
uses the same path, seed, plant draw, current, and disturbance sequence. The
baseline is the zero-action DDAT environment controller.
"""

import argparse
import csv
import json
import os
from pathlib import Path

# The MPC solves many small systems; avoid oversubscribing BLAS threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/uuv_matplotlib")

import numpy as np
import torch

from trajectories.selection import select_paths
from evaluation.metrics import summarize_paths
from research.variants import create_environment, create_network
from simulator.batch_path_env import PathBank
from simulator.model import LoopPath


ROOT = Path(__file__).resolve().parents[1]

# Validation settings. None selects the checkpoint or training-config default.
CHECKPOINT_PATH = None  # newest DDAT or experiments checkpoint
VALIDATION_PATH = "trajectories/stress_15mps/cold_helix_r49_15p9mps.csv" # optionally select one CSV; default is screened val set
EVAL_EPISODES = None  # maximum number of val paths
DEVICE = "cpu"  # uses CUDA if available, otherwise CPU
SEED = None  # defaults to the training seed + 100_000
EVAL_SEEDS = 1  # same one-seed-per-path default as DDAT_train.py evaluation
RANDOMIZE_PLANT = None  # None uses the checkpoint training setting
FEASIBLE_ONLY = None  # None uses the checkpoint training setting
OUTPUT_DIR = None  # e.g. ROOT / "outputs/DDAT/validation"
NO_SHOW = False


def latest_checkpoint():
    candidates = list((ROOT / "outputs/DDAT").glob("*/final_actor.pt"))
    candidates.extend((ROOT / "outputs/experiments").glob("*/*/final_actor.pt"))
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def load_actor(checkpoint_path, device, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "DDAT_UUV_current":
        raise ValueError("checkpoint is not a current-aware DDAT model")
    if "config" not in checkpoint or "agent" not in checkpoint:
        raise ValueError("checkpoint is missing its training config or agent state")
    actor = create_network(checkpoint["config"], checkpoint["observation_dim"], device)
    actor.load_state_dict(checkpoint["agent"])
    actor.eval()
    return actor, checkpoint


def controller_config(saved):
    """Recover the exact controller switches recorded at training time.

    Older checkpoints lack these switches; their historical defaults match
    DDAT_train.evaluate_actor. Current checkpoints supply every value.
    """
    config = dict(saved)
    for name in ("horizon", "backend", "admm_iterations", "adaptation_time",
                 "current_max_mps", "current_filter_time", "seed"):
        if name not in config:
            raise ValueError(f"checkpoint training config is missing {name}")
    legacy_defaults = {
        "admm_rho": 2., "attitude_pitch_weight": 0.,
        "attitude_yaw_weight": 0., "fsp_enabled": False,
        "fsp_filter_time": .4, "fsp_preview_time": None,
        "preview_mode": "dense", "yaw_reward": "heading",
        "randomize_plant": False, "feasible_only": False,
        "eval_episodes": 16, "torch_threads": 2,
    }
    for name, value in legacy_defaults.items():
        config.setdefault(name, value)
    return config


def evaluation_device(requested, backend):
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu")
        if requested == "cuda" and backend in ("osqp", "rti_cpu"):
            raise ValueError(f"{backend} requires --device cpu")
        return requested
    if backend in ("osqp", "rti_cpu"):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def validation_paths(checkpoint, checkpoint_path, path_override, eval_episodes,
                     feasible_only, randomize_plant, split="val"):
    if path_override is not None:
        if not path_override.is_file():
            raise FileNotFoundError(f"trajectory not found: {path_override}")
        return [path_override]
    configured_dir = checkpoint["config"].get(
        "path_dir", checkpoint["config"].get("corpus_dir", "DDAT_trajectories"))
    path_dir = Path(configured_dir)
    if path_dir.is_absolute():
        bundled_dir = ROOT / path_dir.name
        if bundled_dir.is_dir():
            path_dir = bundled_dir
    else:
        path_dir = ROOT / path_dir
    if split == "train":
        manifest = checkpoint_path.parent / "training_paths.json"
        if manifest.is_file():
            recorded = [Path(name) for name in json.loads(manifest.read_text())["paths"]]
            files = []
            for file in recorded:
                bundled_train = path_dir / "train" / file.name
                bundled_root = path_dir / file.name
                files.append(bundled_train if bundled_train.is_file() else
                             bundled_root if bundled_root.is_file() else file)
        else:
            files = sorted((path_dir / "train").glob("*.csv")) + sorted(path_dir.glob("*.csv"))
            files, _ = select_paths(files, feasible_only, randomize_plant)
        if not files or any(not file.is_file() for file in files):
            raise FileNotFoundError(f"training paths missing for {checkpoint_path}")
    else:
        files = sorted((path_dir / "val").glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no validation CSV paths found in {path_dir / 'val'}")
        files, _ = select_paths(files, feasible_only, randomize_plant)
    count = min(eval_episodes, len(files))
    indices = np.linspace(0, len(files) - 1, count, dtype=int)
    return [files[index] for index in indices]


@torch.no_grad()
def run_episode(bank, actor, config, device, seed, randomize_plant, label):
    env = create_environment(config, bank, 1, device, seed=seed,
                             randomize_plant=randomize_plant)
    observation = env.reset(torch.zeros(1, device=env.device, dtype=torch.long))
    current_ned = env.current_ned[0].cpu().tolist()
    rows = []
    episode_return = 0.0
    while True:
        action = (actor.deterministic(observation) if actor is not None else
                  torch.zeros(1, 3, device=env.device))
        base_command = env.base_command[0].clone()
        observation, reward, done, info = env.step(action)
        episode_return += reward[0].item()
        progress = env.progress[0].item()
        reference = bank.sample(env.path_ids, env.progress[:, None])[0][0, 0]
        _, lateral, depth, _, _, speed, _ = env._errors()
        state = env.state[0].cpu().tolist()
        rows.append({
            "step": env.steps[0].item(),
            "time_s": env.steps[0].item() * env.mpc.dt,
            "progress_m": progress,
            "progress_fraction": info["progress_fraction"][0].item(),
            "north_m": state[0], "east_m": state[1], "depth_m": state[2],
            "reference_north_m": reference[0].item(),
            "reference_east_m": reference[1].item(),
            "reference_depth_m": reference[2].item(),
            "position_error_m": info["position_error_m"][0].item(),
            "lateral_error_m": lateral[0].item(),
            "depth_error_m": depth[0].item(),
            "speed_error_mps": speed[0].item(),
            "pitch_deg": float(np.rad2deg(state[3])),
            "body_speed_mps": float(np.linalg.norm(state[5:8])),
            "reward": reward[0].item(),
            "action_thrust": action[0, 0].item(),
            "action_pitch_rate": action[0, 1].item(),
            "action_yaw_rate": action[0, 2].item(),
            "mpc_thrust_n": base_command[0].item(),
            "mpc_elevator_rad": base_command[1].item(),
            "mpc_rudder_rad": base_command[2].item(),
            "command_thrust_n": info["command"][0, 0].item(),
            "command_elevator_rad": info["command"][0, 1].item(),
            "command_rudder_rad": info["command"][0, 2].item(),
            "mpc_ok": int(info["mpc_ok"][0].item()),
            "mpc_qp_valid": int(info["mpc_qp_valid"][0].item()),
            "mpc_failure_streak": int(info["mpc_failure_streak"][0].item()),
            "mpc_primal_residual": info["mpc_primal_residual"][0].item(),
            "mpc_dual_residual": info["mpc_dual_residual"][0].item(),
        })
        if done[0].item():
            break

    def column(name):
        return np.asarray([row[name] for row in rows])

    summary = {
        "policy": label,
        "success": int(info["success"][0].item()),
        "failed": int(info["failed"][0].item()),
        "timeout": int(info["timeout"][0].item()),
        "safe_success": int(info["success"][0].item() and
                            not info["failed"][0].item()),
        "steps": len(rows),
        "duration_s": rows[-1]["time_s"],
        "return": episode_return,
        "progress_fraction": rows[-1]["progress_fraction"],
        "position_rmse_m": float(np.sqrt(np.mean(column("position_error_m") ** 2))),
        "position_max_error_m": float(column("position_error_m").max()),
        "lateral_rmse_m": float(np.sqrt(np.mean(column("lateral_error_m") ** 2))),
        "depth_rmse_m": float(np.sqrt(np.mean(column("depth_error_m") ** 2))),
        "max_abs_pitch_deg": float(np.abs(column("pitch_deg")).max()),
        "max_body_speed_mps": float(column("body_speed_mps").max()),
        "endpoint_error_m": float(np.linalg.norm(
            np.asarray(state[:3]) - bank.points[0, -1].cpu().numpy())),
        "mpc_fallback_fraction": float(1 - column("mpc_ok").mean()),
        "current_north_mps": current_ned[0],
        "current_east_mps": current_ned[1],
    }
    return rows, summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup_plotting(no_show):
    import matplotlib
    import mpl_toolkits

    # Match mplot3d to Matplotlib in environments with two toolkit installs.
    toolkit_dir = Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
    if (toolkit_dir / "mplot3d").is_dir():
        mpl_toolkits.__path__ = [str(toolkit_dir)] + [
            location for location in mpl_toolkits.__path__
            if location != str(toolkit_dir)]
    matplotlib.use("Agg" if no_show else "TkAgg")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.pyplot as plt
    return plt


def plot_tracking(plt, output, path_file, path, runs, step, seed):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    horizontal, depth_ax, error_ax, action_ax = axes.flat
    horizontal.plot(path.points[:, 1], path.points[:, 0], "k--", label="Reference")
    depth_ax.plot(path.distances, path.points[:, 2], "k--", label="Reference")
    for label, rows, summary, color in runs:
        time = [row["time_s"] for row in rows]
        east = [row["east_m"] for row in rows]
        north = [row["north_m"] for row in rows]
        horizontal.plot(east, north, color=color, label=f"{label} ({'success' if summary['success'] else 'stopped'})")
        horizontal.plot(east[-1], north[-1], "o", color=color)
        depth_ax.plot([row["progress_m"] for row in rows],
                      [row["depth_m"] for row in rows], color=color, label=label)
        error_ax.plot(time, [row["position_error_m"] for row in rows],
                      color=color, label=label)
        if label == "DDAT + MPC":
            for field, name in (("action_thrust", "Thrust"),
                                ("action_pitch_rate", "Pitch rate"),
                                ("action_yaw_rate", "Yaw rate")):
                action_ax.plot(time, [row[field] for row in rows], label=name)
    horizontal.set(xlabel="East (m)", ylabel="North (m)", title="Horizontal trajectory")
    horizontal.set_aspect("equal", adjustable="datalim")
    depth_ax.set(xlabel="Path progress (m)", ylabel="Depth (m)", title="Depth tracking")
    depth_ax.invert_yaxis()
    error_ax.set(xlabel="Time (s)", ylabel="Position error (m)", title="Tracking error")
    action_ax.set(xlabel="Time (s)", ylabel="Normalized DDAT action",
                  title="Policy correction", ylim=(-1.05, 1.05))
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"{path_file.name} | training step {step} | seed {seed}")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_trajectories(plt, output, path_file, path, runs, step, seed):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot(path.points[:, 1], path.points[:, 0], path.points[:, 2],
            "k--", label="Reference")
    positions = [path.points]
    for label, rows, summary, color in runs:
        points = np.asarray([[row["north_m"], row["east_m"], row["depth_m"]]
                             for row in rows])
        points = np.vstack((path.points[0], points))
        positions.append(points)
        ax.plot(points[:, 1], points[:, 0], points[:, 2], color=color,
                label=f"{label} ({'success' if summary['success'] else 'stopped'})")
        ax.scatter(*points[-1, [1, 0, 2]], color=color, s=35)
    ax.scatter(*path.points[0, [1, 0, 2]], color="black", marker="D", s=35,
               label="Start")
    all_points = np.vstack(positions)
    minimum, maximum = all_points.min(axis=0), all_points.max(axis=0)
    spans = np.maximum(maximum - minimum, 1.0)
    center = (minimum + maximum) / 2
    for setter, index in ((ax.set_xlim, 1), (ax.set_ylim, 0), (ax.set_zlim, 2)):
        setter(center[index] - .52 * spans[index], center[index] + .52 * spans[index])
    ax.set_box_aspect(tuple(spans[[1, 0, 2]]))
    ax.set(xlabel="East (m)", ylabel="North (m)",
           title=f"{path_file.name} | training step {step} | seed {seed}")
    ax.invert_zaxis()
    ax.view_init(elev=45, azim=-65)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--path", type=Path, default=VALIDATION_PATH,
                        help="validate one trajectory instead of the selected split")
    parser.add_argument("--all-paths", action="store_true",
                        help="use the selected split instead of the default single path")
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--seeds", type=int, default=EVAL_SEEDS,
                        help="paired random seeds per path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="first evaluation seed; default is training seed + 100000")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES,
                        help="maximum number of paths; default is the training setting")
    parser.add_argument("--randomize-plant", action=argparse.BooleanOptionalAction,
                        default=RANDOMIZE_PLANT,
                        help="default follows the checkpoint training setting")
    parser.add_argument("--feasible-only", action=argparse.BooleanOptionalAction,
                        default=FEASIBLE_ONLY,
                        help="default follows the checkpoint training setting")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-show", dest="no_show", action="store_true",
                        default=NO_SHOW)
    parser.add_argument("--show", dest="no_show", action="store_false",
                        help="open the 3D plot for one path and one seed")
    arguments = parser.parse_args(argv)
    checkpoint_path = (Path(arguments.checkpoint) if arguments.checkpoint is not None
                       else latest_checkpoint())
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "DDAT_UUV_current" or "config" not in checkpoint:
        raise ValueError("checkpoint is not a current-aware DDAT model with training config")
    config = controller_config(checkpoint["config"])
    trained_backend = config["backend"]
    device = evaluation_device(arguments.device, trained_backend)
    if device == "cpu" and trained_backend == "rti_cuda":
        config["backend"] = "rti_cpu"
    actor, checkpoint = load_actor(checkpoint_path, device, checkpoint)
    randomize_plant = (config["randomize_plant"] if arguments.randomize_plant is None
                       else arguments.randomize_plant)
    feasible_only = (config["feasible_only"] if arguments.feasible_only is None
                     else arguments.feasible_only)
    eval_episodes = (arguments.eval_episodes if arguments.eval_episodes is not None else
                     config["eval_episodes"])
    if eval_episodes < 1 or arguments.seeds < 1:
        raise ValueError("EVAL_EPISODES and EVAL_SEEDS must be positive")
    path_override = (None if arguments.all_paths else
                     Path(arguments.path) if arguments.path is not None else None)
    paths = validation_paths(checkpoint, checkpoint_path, path_override, eval_episodes,
                             feasible_only, randomize_plant, arguments.split)
    seed = arguments.seed if arguments.seed is not None else config["seed"] + 100_000
    if seed < 0:
        raise ValueError("SEED must be nonnegative")
    torch.set_num_threads(config["torch_threads"])
    selection = (f"single_{paths[0].stem}" if path_override is not None else
                 f"{arguments.split}_{len(paths)}paths")
    output_dir = (Path(arguments.output_dir) if arguments.output_dir is not None else
                  checkpoint_path.parent / "evaluation" /
                  f"{selection}_{arguments.seeds}seeds_seed{seed}_{device}"
                  f"{'_randomized' if randomize_plant else ''}")
    output_dir.mkdir(parents=True, exist_ok=True)

    no_show = (arguments.no_show or len(paths) != 1 or arguments.seeds != 1 or
               not os.environ.get("DISPLAY"))
    try:
        plt = setup_plotting(no_show)
    except ImportError as error:
        raise RuntimeError(f"cannot open a 3D window: {error}; set NO_SHOW = True") from error
    summaries = []
    for path_index, path_file in enumerate(paths):
        # Separate environments let every policy/baseline pair receive the
        # same random sequence. Use distinct seeds for each path and repeat.
        bank = PathBank([path_file], device)
        path = LoopPath(path_file)
        for seed_index in range(arguments.seeds):
            path_seed = seed + path_index*arguments.seeds + seed_index
            path_output = output_dir / path_file.stem / f"seed_{path_seed}"
            path_output.mkdir(parents=True, exist_ok=True)
            runs = []
            for label, policy, color, filename in (
                ("MPC", None, "tab:gray", "mpc_steps.csv"),
                ("DDAT + MPC", actor, "tab:blue", "ddat_mpc_steps.csv"),
            ):
                rows, summary = run_episode(bank, policy, config,
                                            device, path_seed, randomize_plant, label)
                summary.update({"path": str(path_file.resolve()),
                                "checkpoint": str(checkpoint_path.resolve()),
                                "training_step": checkpoint["training_step"],
                                "seed": path_seed, "device": device,
                                "trained_backend": trained_backend,
                                "eval_backend": config["backend"],
                                "randomize_plant": int(randomize_plant),
                                "feasible_only": int(feasible_only),
                                "preview_mode": config["preview_mode"],
                                "yaw_reward": config["yaw_reward"],
                                "attitude_pitch_weight": config["attitude_pitch_weight"],
                                "attitude_yaw_weight": config["attitude_yaw_weight"],
                                "admm_rho": config["admm_rho"],
                                "fsp_enabled": config["fsp_enabled"],
                                "fsp_filter_time": config["fsp_filter_time"],
                                "fsp_preview_time": config["fsp_preview_time"]})
                write_csv(path_output / filename, rows)
                runs.append((label, rows, summary, color))
                summaries.append(summary)
                status = ("success" if summary["safe_success"] else
                          "failed" if summary["failed"] else "timeout")
                print(f"{path_file.name} seed={path_seed} | {label}: {status}, "
                      f"progress={summary['progress_fraction']:.1%}, "
                      f"return={summary['return']:.1f}, "
                      f"position RMSE={summary['position_rmse_m']:.2f} m",
                      flush=True)
            plot_tracking(plt, path_output / "tracking.png", path_file, path, runs,
                          checkpoint["training_step"], path_seed)
            figure = plot_trajectories(
                plt, path_output / "trajectory_comparison.png", path_file, path, runs,
                checkpoint["training_step"], path_seed)
            if no_show:
                plt.close(figure)

    write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    aggregate = {"training_step": checkpoint["training_step"],
                 "split": arguments.split,
                 "paths": [str(path) for path in paths],
                 "seeds_per_path": arguments.seeds,
                 "trained_backend": trained_backend,
                 "eval_backend": config["backend"],
                 "device": device,
                 "randomize_plant": randomize_plant,
                 "feasible_only": feasible_only,
                 "preview_mode": config["preview_mode"],
                 "yaw_reward": config["yaw_reward"],
                 "attitude_pitch_weight": config["attitude_pitch_weight"],
                 "attitude_yaw_weight": config["attitude_yaw_weight"],
                 "admm_rho": config["admm_rho"],
                 "fsp_enabled": config["fsp_enabled"],
                 "fsp_filter_time": config["fsp_filter_time"],
                 "fsp_preview_time": config["fsp_preview_time"]}
    for label, key in (("DDAT + MPC", "policy"), ("MPC", "mpc_only")):
        matching = [row for row in summaries if row["policy"] == label]
        aggregate[key] = summarize_paths(matching)
        completed = [row for row in matching if row["safe_success"]]
        aggregate[key]["safe_successes"] = len(completed)
        aggregate[key]["success_rate"] = len(completed)/len(matching)
        aggregate[key]["mean_position_rmse_m_all_episodes"] = aggregate[key].pop(
            "mean_position_rmse_m")
        aggregate[key]["mean_position_rmse_m_safe_success"] = (
            float(np.mean([row["position_rmse_m"] for row in completed]))
            if completed else None)
    paired = {}
    for row in summaries:
        paired.setdefault((row["path"], row["seed"]), {})[row["policy"]] = row
    both_completed = [pair for pair in paired.values()
                      if pair["MPC"]["safe_success"] and pair["DDAT + MPC"]["safe_success"]]
    aggregate["paired_both_safe_successes"] = len(both_completed)
    aggregate["mean_ddat_minus_mpc_rmse_m_both_safe"] = (
        float(np.mean([pair["DDAT + MPC"]["position_rmse_m"] -
                       pair["MPC"]["position_rmse_m"] for pair in both_completed]))
        if both_completed else None)
    aggregate["rmse_note"] = (
        "all-episode RMSE mixes different failure times; compare RMSE only when both complete safely")
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Results saved to {output_dir}")
    if not no_show:
        print("Close the 3D window to finish.")
        plt.show(block=True)


if __name__ == "__main__":

    main()
