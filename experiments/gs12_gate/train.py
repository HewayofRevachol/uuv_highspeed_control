"""On-policy four-channel PPO; every update consumes one policy version."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.gs12_gate.data_io import file_hash, immutable_json, immutable_jsonl, read_json
from experiments.gs12_gate.episode import run_episode
from experiments.gs12_gate.features import GateActorCritic
from experiments.gs12_gate.run import _artifacts, _load_run
from training.ppo import advantages, ppo_update


def transitions(path):
    with np.load(path/"decisions.npz") as rows:
        return [{key: rows[key][i] for key in rows.files} for i in range(len(rows["t"]))]


def _batch(rows):
    rewards = torch.tensor(np.asarray([x["reward"] for x in rows]), dtype=torch.float32)
    values = torch.tensor(np.asarray([x["value"] for x in rows]), dtype=torch.float32)
    next_values = torch.tensor(np.asarray([x["next_value"] for x in rows]), dtype=torch.float32)
    terminated = torch.tensor([bool(x["terminal"]) for x in rows])
    done = terminated | torch.tensor([bool(x["truncated"]) for x in rows])
    adv, returns = advantages(rewards, values, next_values, done, terminated, gamma=1., lam=.95)
    return (torch.tensor(np.asarray([x["observation"] for x in rows]), dtype=torch.float32),
            torch.tensor(np.asarray([x["raw_z"] for x in rows]), dtype=torch.float32),
            torch.tensor(np.asarray([x["log_prob"] for x in rows]), dtype=torch.float32),
            returns, adv)


def _checkpoint(path, policy, optimizer, rng, seed, decisions, updates, run,
                resolved_cases_hash):
    if path.exists(): raise FileExistsError(path)
    payload = {"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
               "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
               "seed": seed, "decisions": decisions, "updates": updates,
               "policy_variant": policy.variant,
               "resolved_train_cases_hash": resolved_cases_hash,
               "normalization_hash": file_hash(Path(run)/"normalization.json"),
               "residual_hash": file_hash(Path(run)/"residual_scale.json"),
               "schema_hash": file_hash(Path(run)/"obs_schema.json")}
    torch.save(payload, path)
    return file_hash(path)


def _resolved_train_cases(run, cases, direction_source_path=None):
    pending = [case for case in cases if case["split"] == "train" and
               case.get("direction_class") == "svd" and
               case.get("complex_direction") is None]
    source_hash = None
    vector = None
    if pending:
        source_path = (Path(direction_source_path) if direction_source_path else
                       Path(run)/"frequency"/"dev_direction"/"direction_source.json")
        if not source_path.exists():
            raise FileNotFoundError("frozen dev SVD direction required before training; "
                                    "run frequency --mode freeze-direction")
        source = read_json(source_path)
        if source.get("split") != "dev" or source.get("frequency_bin") != 6:
            raise ValueError("training source must be development 64 s bin 6")
        vector = np.asarray(source["right_singular_vector"], float)
        if (vector.shape != (3, 2) or not np.isfinite(vector).all() or
                not np.isclose(np.linalg.norm(vector), 1., atol=1e-6)):
            raise ValueError("malformed frozen complex direction")
        source_hash = file_hash(source_path)
    resolved = []
    for case in cases:
        if case["split"] != "train": continue
        item = dict(case)
        if item.get("direction_class") == "svd" and item.get("complex_direction") is None:
            item["complex_direction"] = vector.tolist()
            item["direction_source_hash"] = source_hash
        resolved.append(item)
    path = Path(run)/"train_cases_resolved.jsonl"
    return resolved, immutable_jsonl(path, resolved)


def _select_dev(run, seed, checkpoints, artifacts, variant):
    _, cases = _load_run(run)
    dev = [c for c in cases if c["split"] == "dev_quick"]
    scores = {}
    for path in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["policy_variant"] != variant:
            raise ValueError("checkpoint policy variant mismatch")
        policy = GateActorCritic(variant); policy.load_state_dict(saved["policy"]); policy.eval()
        rows = []
        for case in dev:
            target = (Path(run)/"episodes"/"dev_checkpoint"/variant/str(seed)/
                      path.stem/case["case_id"])
            rows.append(run_episode(case, "ppo_mean", target, artifacts=artifacts, policy=policy))
        scores[path.name] = {"failures": sum(not r["complete"] for r in rows),
                             "mean_cost": float(np.mean([r["tracking_cost_integral"] for r in rows]))}
    chosen = min(checkpoints, key=lambda p: (scores[p.name]["failures"], scores[p.name]["mean_cost"]))
    result = {"seed": seed, "variant": variant,
              "selected": chosen.name, "scores": scores,
              "selection_split": "dev_quick"}
    selected_dir = (Path(run)/"training"/str(seed) if variant == "standard" else
                    Path(run)/"training"/"ablations"/variant/str(seed))
    immutable_json(selected_dir/"selected_checkpoint.json", result)
    return result


def train_one(run, seed=41001, decisions=8192, update_every=512,
              select_dev=True, variant="standard", direction_source_path=None):
    if decisions <= 0 or update_every <= 0 or decisions % update_every:
        raise ValueError("decisions must be a positive multiple of update_every")
    manifest, cases = _load_run(run)
    train_cases, resolved_cases_hash = _resolved_train_cases(run, cases,
                                                             direction_source_path)
    if not train_cases: raise ValueError("no training cases")
    artifacts = _artifacts(run)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    policy = GateActorCritic(variant)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    target = (Path(run)/"training"/str(seed) if variant == "standard" else
              Path(run)/"training"/"ablations"/variant/str(seed))
    if variant != "standard":
        immutable_json(Path(run)/"training"/"ablations"/variant/"ablation_protocol.json",
            {"variant": variant, "decisions": decisions, "update_every": update_every,
             "policy_input_dim": GateActorCritic.VARIANTS[variant][0],
             "policy_action_dim": GateActorCritic.VARIANTS[variant][1],
             "normalization_hash": file_hash(Path(run)/"normalization.json"),
             "schema_hash": file_hash(Path(run)/"obs_schema.json"),
             "residual_hash": file_hash(Path(run)/"residual_scale.json"),
             "resolved_train_cases_hash": resolved_cases_hash,
             "selection_split": "dev_quick"})
    if target.exists(): raise FileExistsError(target)
    target.mkdir(parents=True)
    updates_file = target/"updates.jsonl"
    total_control = total_substeps = episode_index = 0
    checkpoints = []
    started = time.perf_counter()
    with updates_file.open("w", encoding="utf-8") as stream:
        for update in range(decisions//update_every):
            samples = []
            while len(samples) < update_every:
                # Choosing the case before rollout uses no dev or test metadata.
                case = train_cases[int(rng.integers(len(train_cases)))]
                remaining = update_every-len(samples)
                duration = min(case["duration_s"], remaining*.5)
                episode_path = (Path(run)/"episodes"/"train"/variant/str(seed)/
                                f"{episode_index:05d}")
                summary = run_episode(case, "ppo", episode_path,
                                      artifacts=artifacts, policy=policy,
                                      random_seed=int(rng.integers(2**31)),
                                      max_seconds=duration)
                new_rows = transitions(episode_path)
                if len(new_rows) > remaining:
                    raise AssertionError("rollout exceeded on-policy batch boundary")
                samples.extend(new_rows)
                total_control += summary["control_steps"]
                total_substeps += summary["substeps"]
                episode_index += 1
            metrics = ppo_update(policy, optimizer, _batch(samples), epochs=4,
                                 minibatch=64, clip=.2, value_weight=.5,
                                 entropy_weight=.01, grad_norm=1.)
            cumulative = (update+1)*update_every
            checkpoint_hash = None
            if cumulative in (decisions//2, decisions):
                checkpoint = target/f"checkpoint_{cumulative:06d}.pt"
                checkpoint_hash = _checkpoint(checkpoint, policy, optimizer,
                    rng, seed, cumulative, update+1, run, resolved_cases_hash)
                checkpoints.append(checkpoint)
            entry = {"seed": seed, "variant": variant, "update": update+1,
                     "decisions": cumulative, "control_steps": total_control,
                     "substeps": total_substeps, "episodes": episode_index,
                     "checkpoint_hash": checkpoint_hash,
                     "wall_s": time.perf_counter()-started, **metrics}
            stream.write(json.dumps(entry, ensure_ascii=False)+"\n")
            stream.flush()
    selected = _select_dev(run, seed, checkpoints, artifacts, variant) if select_dev else None
    result = {"seed": seed, "variant": variant,
              "resolved_train_cases_hash": resolved_cases_hash,
              "decisions": decisions, "updates": decisions//update_every,
              "control_steps": total_control, "substeps": total_substeps,
              "episodes": episode_index, "wall_s": time.perf_counter()-started,
              "checkpoints": [p.name for p in checkpoints], "dev_selection": selected}
    immutable_json(target/"training_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, default=41001)
    group.add_argument("--seeds")
    parser.add_argument("--decisions", type=int, default=8192)
    parser.add_argument("--update-every", type=int, default=512)
    parser.add_argument("--no-dev-selection", action="store_true")
    parser.add_argument("--ablation", choices=("tied_residual", "no_history"))
    parser.add_argument("--direction-source", type=Path)
    args = parser.parse_args()
    if args.config is not None and read_json(args.config) != read_json(args.run/"run_manifest.json")["config"]:
        raise ValueError("training config differs from prepared manifest")
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed]
    results = [train_one(args.run, seed, args.decisions, args.update_every,
                         select_dev=not args.no_dev_selection,
                         variant=args.ablation or "standard",
                         direction_source_path=args.direction_source) for seed in seeds]
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
