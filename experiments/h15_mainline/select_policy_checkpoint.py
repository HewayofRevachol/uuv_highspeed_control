"""Freeze one H15 masked-PPO checkpoint using only the paired dev split."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.h15_mainline.train_masked_ppo import POLICY_SEEDS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = []
    identity = None
    for directory in args.runs:
        run = json.loads((directory / "run.json").read_text())
        selection = json.loads((directory / "selection.json").read_text())
        if run["scope"] != "frozen train/dev" or run["policy_seed"] not in POLICY_SEEDS:
            raise ValueError(f"not a formal frozen training run: {directory}")
        shared = tuple(run[key] for key in ("freeze_sha256",
                                           "estimator_checkpoint_sha256",
                                           "candidate_config_sha256",
                                           "reward_config_sha256"))
        if identity is None:
            identity = shared
        elif shared != identity:
            raise ValueError("the three policy seeds used different frozen inputs")
        if selection["selected"] not in ("initial", "trained"):
            raise ValueError("unexpected within-seed checkpoint selection")
        checkpoint = Path(selection["checkpoint"])
        if sha(checkpoint) != selection["checkpoint_sha256"]:
            raise ValueError("selected checkpoint hash differs")
        scores = json.loads((directory / "dev_scores.json").read_text())
        if scores != selection["dev_scores"] or any(
                scores[name]["episodes"] != 6 for name in ("initial", "trained")):
            raise ValueError("incomplete or changed dev selection evidence")
        rows.append({"seed": run["policy_seed"], "run": str(directory),
                     "run_sha256": sha(directory / "run.json"),
                     "selection_sha256": sha(directory / "selection.json"),
                     "checkpoint": str(checkpoint),
                     "checkpoint_sha256": selection["checkpoint_sha256"],
                     "within_seed_choice": selection["selected"],
                     "dev_score": scores[selection["selected"]]["score"]})
    if sorted(row["seed"] for row in rows) != list(POLICY_SEEDS):
        raise ValueError("exactly the three frozen policy seeds are required")
    selected = max(rows, key=lambda row: (tuple(row["dev_score"]), -row["seed"]))
    result = {"schema": 1, "method": "H15_dev_only_masked_PPO_selection",
              "selected_seed": selected["seed"],
              "checkpoint": selected["checkpoint"],
              "checkpoint_sha256": selected["checkpoint_sha256"],
              "selection_rule": "lexicographic frozen dev score; smaller policy seed breaks ties",
              "shared_input_sha256": dict(zip(("freeze", "estimator", "candidate",
                                               "reward"), identity)),
              "seed_results": sorted(rows, key=lambda row: row["seed"]),
              "test_used": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_seed": selected["seed"],
                      "dev_score": selected["dev_score"],
                      "checkpoint_sha256": selected["checkpoint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
