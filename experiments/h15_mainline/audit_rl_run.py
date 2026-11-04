"""Audit H15 B3 masked decisions against logged candidate availability."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.h15_mainline.cycle_log import validate as validate_cycle
from experiments.h15_mainline.masked_speed_ppo import GRID
from experiments.h15_mainline.run_record import validate as validate_run


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory):
    summary = json.loads((directory / "summary.json").read_text())
    run = validate_run(json.loads((directory / "run.json").read_text()))
    mode = json.loads((directory / "policy_mode.json").read_text())
    cycles = [validate_cycle(json.loads(line)) for line in
              (directory / "cycle_log.jsonl").read_text().splitlines()]
    decisions = [json.loads(line) for line in
                 (directory / "policy_decisions.jsonl").read_text().splitlines()]
    substeps = json.loads((directory / "substep_samples.json").read_text())
    if (run["method"] != "B3_RL" or summary["method"] != "B3_RL" or
            mode["action_grid_mps"] != list(GRID) or
            mode["decision_periods"] != 5 or
            sha(ROOT / mode["checkpoint"]) != mode["checkpoint_sha256"] or
            sha(ROOT / mode["reward_config"]) != mode["reward_config_sha256"] or
            run["config_sha256"]["policy_mode.json"] != sha(directory / "policy_mode.json") or
            len(cycles) != summary["periods"] or len(substeps) != 5*len(cycles) or
            len(decisions) != summary["policy_decision_count"]):
        raise ValueError("B3 run identity, provenance or trace length differs")
    decision_by_period = {row["period"]: row for row in decisions}
    if len(decision_by_period) != len(decisions):
        raise ValueError("duplicate policy decision period")
    held = 15.
    for period, cycle in enumerate(cycles):
        if cycle["period"] != period:
            raise ValueError("nonconsecutive cycle log")
        mask = [item["reason"] is None for item in cycle["candidate_rejections"]]
        if len(mask) != len(GRID) or any(
                item["candidate_mps"] != speed for item, speed in
                zip(cycle["candidate_rejections"], GRID)):
            raise ValueError("candidate grid differs")
        decision = decision_by_period.get(period)
        if decision is not None:
            if (period % 5 or not any(mask) or decision["mask"] != mask or
                    not mask[decision["action_index"]] or
                    decision["intent_mps"] != GRID[decision["action_index"]] or
                    not all(math.isfinite(decision[key]) for key in
                            ("reward", "logp", "value"))):
                raise ValueError(f"invalid masked decision at {period}")
            held = decision["intent_mps"]
        elif period % 5 == 0 and any(mask):
            raise ValueError(f"missing policy decision at {period}")
        if cycle["raw_rl_intent_mps"] != held:
            raise ValueError(f"held intent changed outside policy decision at {period}")
        selected = cycle["selected_candidate_mps"]
        if selected is not None and (selected != held or not mask[GRID.index(selected)]):
            raise ValueError(f"selected intent was not verified at {period}")
        if selected is None and cycle["fallback"] is None:
            raise ValueError(f"missing fallback at {period}")
    if decisions and not decisions[-1]["terminal"]:
        raise ValueError("last decision lacks terminal marker")
    return {"periods": len(cycles), "substeps": len(substeps),
            "decisions": len(decisions),
            "empty_cycles": sum(c["candidate_empty"] for c in cycles),
            "backup_cycles": sum(c["fallback"] == "one_step_verified_fixed15"
                                 for c in cycles),
            "completed": summary["completed"],
            "strict_mainline": summary["strict_mainline"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = audit(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
