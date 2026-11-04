"""Train three-action-seed H15 masked PPO policies on frozen train cases.

Each policy rollout uses the same candidate MPC, nonlinear screen and backup
as B0/B2. Dev evaluation selects between the initial and trained checkpoints.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments.h15_mainline.masked_speed_ppo import MaskedSpeedActorCritic


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCHEDULE = (("train_turn_wide", 31001),
                  ("train_reversal_wide", 31002),
                  ("train_coupled_shallow", 31003),
                  ("train_late_entry", 31001))
POLICY_SEEDS = (41001, 41002, 41003)
PPO = {"learning_rate": 3e-4, "epochs_per_episode": 4,
       "clip_ratio": .2, "value_weight": .5, "entropy_weight": .01,
       "gradient_norm_max": 1.}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(path, model, reward, estimator, candidate, seed, update):
    torch.save({"model_state_dict": model.state_dict(),
                "reward_config": json.loads(Path(reward).read_text()),
                "reward_config_sha256": sha(reward),
                "estimator_checkpoint_sha256": sha(estimator),
                "candidate_config_sha256": sha(candidate),
                "policy_seed": seed, "update": update}, path)


def run_episode(args, split, case_id, plant_seed, method, output, policy_path=None,
                sample=False, policy_seed=0):
    command = ["/bin/python3", "-m",
               "experiments.h15_mainline.run_offline_governor_smoke",
               "--freeze", str(args.freeze), "--checkpoint", str(args.checkpoint),
               "--config", str(args.config), "--split", split,
               "--case-id", case_id, "--plant-seed", str(plant_seed),
               "--method", method, "--backup-mode", "one_step_verified",
               "--estimator-mode", "E1", "--out", str(output)]
    if args.pilot_periods:
        command += ["--max-periods", str(args.pilot_periods)]
    if method == "B3_RL":
        command += ["--policy-checkpoint", str(policy_path),
                    "--reward-config", str(args.reward),
                    "--policy-seed", str(policy_seed)]
        if sample:
            command.append("--policy-sample")
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
    result = subprocess.run(command, cwd=ROOT, env=environment,
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{method} {split}/{case_id}/{plant_seed}: "
                           f"{result.stderr[-2000:]}")
    return json.loads((output / "summary.json").read_text())


def update_ppo(model, optimizer, rows, discount):
    if not rows:
        return {"decisions": 0, "update_skipped": "no_verified_decision"}
    observations = torch.tensor([r["observation"] for r in rows], dtype=torch.float32)
    masks = torch.tensor([r["mask"] for r in rows], dtype=torch.bool)
    actions = torch.tensor([r["action_index"] for r in rows], dtype=torch.long)
    old_logp = torch.tensor([r["logp"] for r in rows], dtype=torch.float32)
    old_value = torch.tensor([r["value"] for r in rows], dtype=torch.float32)
    rewards = [float(r["reward"]) for r in rows]
    returns = []
    running = 0.
    for reward, row in zip(reversed(rewards), reversed(rows)):
        running = reward + discount*running*(not row["terminal"])
        returns.append(running)
    returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
    advantages = returns-old_value
    if len(rows) > 1 and float(advantages.std()) > 1e-8:
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
    last_loss = None
    for _ in range(PPO["epochs_per_episode"]):
        logp, entropy, value = model.evaluate(observations, masks, actions)
        ratio = (logp-old_logp).exp()
        surrogate = torch.minimum(ratio*advantages,
                                  ratio.clamp(1-PPO["clip_ratio"],
                                              1+PPO["clip_ratio"])*advantages)
        loss = (-surrogate.mean() + PPO["value_weight"]*(value-returns).square().mean()
                - PPO["entropy_weight"]*entropy.mean())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), PPO["gradient_norm_max"])
        optimizer.step()
        last_loss = float(loss.detach())
    return {"decisions": len(rows), "mean_reward_per_decision": float(np.mean(rewards)),
            "ppo_loss": last_loss}


def dev_score(summaries):
    return (sum(bool(s["strict_mainline"]) for s in summaries),
            sum(bool(s["completed"]) for s in summaries),
            -sum(s["backup_periods"] for s in summaries),
            sum(s["final_progress_m"] for s in summaries),
            -sum(s["max_abs_lateral_m"] for s in summaries),
            -sum(s["max_abs_speed_error_mps"] for s in summaries))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--policy-seed", type=int, choices=POLICY_SEEDS, required=True)
    parser.add_argument("--pilot-periods", type=int, default=0,
                        help="truncate one diagnostic train episode and skip dev")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.pilot_periods < 0:
        raise ValueError("pilot periods must be nonnegative")
    args.out.mkdir(parents=True)
    frozen = json.loads(args.freeze.read_text())
    for split in ("train", "dev"):
        path = ROOT / frozen["manifests"][split]["path"]
        if sha(path) != frozen["manifests"][split]["sha256"]:
            raise RuntimeError(f"changed {split} manifest")
    if json.loads(args.config.read_text())["estimator_checkpoint_sha256"] != sha(args.checkpoint):
        raise RuntimeError("candidate config and estimator checkpoint differ")
    reward = json.loads(args.reward.read_text())
    if reward["name"] != "h15_masked_speed_ppo_reward_v1":
        raise RuntimeError("unexpected reward version")
    torch.set_num_threads(1)
    torch.manual_seed(args.policy_seed)
    model = MaskedSpeedActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=PPO["learning_rate"])
    initial = args.out / "policy_initial.pt"
    save_checkpoint(initial, model, args.reward, args.checkpoint, args.config,
                    args.policy_seed, 0)
    record = {"schema": 1, "method": "H15_five_action_masked_PPO",
              "scope": "pilot" if args.pilot_periods else "frozen train/dev",
              "policy_seed": args.policy_seed, "policy_seeds_frozen": POLICY_SEEDS,
              "train_schedule": TRAIN_SCHEDULE, "ppo": PPO,
              "freeze_sha256": sha(args.freeze),
              "estimator_checkpoint_sha256": sha(args.checkpoint),
              "candidate_config_sha256": sha(args.config),
              "reward_config_sha256": sha(args.reward),
              "source_sha256": {name: sha(ROOT / "experiments/h15_mainline" / name)
                                for name in ("masked_speed_ppo.py",
                                             "run_offline_governor_smoke.py",
                                             "train_masked_ppo.py")},
              "python_executable": "/bin/python3"}
    (args.out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    history = []
    schedule = TRAIN_SCHEDULE[:1] if args.pilot_periods else TRAIN_SCHEDULE
    for episode_index, (case_id, plant_seed) in enumerate(schedule):
        checkpoint_path = initial if episode_index == 0 else args.out / f"policy_after_{episode_index-1}.pt"
        paths = {}
        summaries = {}
        for method in ("B0_shared", "B2_gov", "B3_RL"):
            output = args.out / f"train_{episode_index}_{case_id}_s{plant_seed}_{method}"
            summaries[method] = run_episode(
                args, "train", case_id, plant_seed, method, output,
                checkpoint_path if method == "B3_RL" else None,
                sample=method == "B3_RL", policy_seed=args.policy_seed+episode_index)
            paths[method] = str(output)
        trace = Path(paths["B3_RL"]) / "policy_decisions.jsonl"
        decisions = [json.loads(line) for line in trace.read_text().splitlines()]
        update = update_ppo(model, optimizer, decisions,
                            reward["discount_per_decision"])
        after = args.out / f"policy_after_{episode_index}.pt"
        save_checkpoint(after, model, args.reward, args.checkpoint, args.config,
                        args.policy_seed, episode_index+1)
        history.append({"episode": episode_index, "case_id": case_id,
                        "plant_seed": plant_seed, "runs": paths,
                        "results": {method: {key: summary[key] for key in
                            ("completed", "strict_mainline", "backup_periods",
                             "final_progress_m", "max_abs_lateral_m",
                             "max_abs_speed_error_mps")}
                            for method, summary in summaries.items()},
                        "ppo": update, "checkpoint_sha256": sha(after)})
        (args.out / "train_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"train {episode_index}: decisions={update['decisions']}", flush=True)
    if args.pilot_periods:
        return
    manifest = json.loads((ROOT / frozen["manifests"]["dev"]["path"]).read_text())
    scores = {}
    for name, checkpoint_path in (("initial", initial),
                                  ("trained", args.out / f"policy_after_{len(schedule)-1}.pt")):
        summaries = []
        for case in manifest["cases"]:
            for plant_seed in manifest["seeds"]:
                output = args.out / f"dev_{name}_{case['id']}_s{plant_seed}_B3_RL"
                summaries.append(run_episode(args, "dev", case["id"], plant_seed,
                                             "B3_RL", output, checkpoint_path))
        scores[name] = {"checkpoint_sha256": sha(checkpoint_path),
                        "score": dev_score(summaries),
                        "episodes": len(summaries)}
        (args.out / "dev_scores.json").write_text(json.dumps(scores, indent=2) + "\n")
        print(f"dev {name}: {scores[name]['score']}", flush=True)
    selected = max(("initial", "trained"), key=lambda name: tuple(scores[name]["score"]))
    selected_path = initial if selected == "initial" else args.out / f"policy_after_{len(schedule)-1}.pt"
    (args.out / "selection.json").write_text(json.dumps(
        {"selected": selected, "checkpoint": str(selected_path),
         "checkpoint_sha256": sha(selected_path), "dev_scores": scores,
         "selection_rule": "strict completion, completion, fewer backups, more progress, lower lateral error, lower speed error; tie to initial"},
        indent=2) + "\n")


if __name__ == "__main__":
    main()
