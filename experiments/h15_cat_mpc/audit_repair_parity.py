"""Validate and summarize the repaired 12-episode A0/A4 parity run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.h15_cat_mpc.residual_predictor import RESIDUAL_LABEL_VERSION


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    episodes, manifests = [], []
    for episode_path in sorted(args.runs.glob("*/episode.json")):
        episode = json.loads(episode_path.read_text())
        run = json.loads((episode_path.parent/"run.json").read_text())
        if episode.get("command_schema") != "issued_v2" or episode.get("schema") != 2:
            raise ValueError(f"legacy episode schema: {episode_path}")
        if episode.get("residual_label_version") != RESIDUAL_LABEL_VERSION:
            raise ValueError(f"unexpected residual labels: {episode_path}")
        if not episode.get("target_domain_applied_after_reset"):
            raise ValueError(f"target domain was not applied in reset: {episode_path}")
        snapshot = episode["target_domain_reset_snapshot"]
        before = np.asarray(snapshot["parameters_fin_before"], dtype=float)
        after = np.asarray(snapshot["parameters_fin_after"], dtype=float)
        if not np.allclose(after, before*.9, rtol=0., atol=1e-12):
            raise ValueError(f"target fin gain snapshot mismatch: {episode_path}")
        if run.get("target_domain_applied_after_reset") is not True:
            raise ValueError(f"run manifest omits reset-hook confirmation: {episode_path}")
        rows = episode["completed_rows"]
        if len(rows) != len(episode["trace"]):
            raise ValueError(f"period/log count differs: {episode_path}")
        for row, trace in zip(rows, episode["trace"]):
            if row.get("command_schema") != "issued_v2":
                raise ValueError(f"legacy command row: {episode_path}")
            mpc = np.asarray(json.loads(row["mpc_command_json"]))
            compensation = np.asarray(json.loads(row["low_level_compensation_json"]))
            issued = np.asarray(json.loads(row["issued_command_json"]))
            compat = np.asarray(json.loads(row["command_json"]))
            if not (np.allclose(issued, mpc+compensation, atol=1e-6, rtol=0.) and
                    np.allclose(issued, compat, atol=1e-6, rtol=0.) and
                    np.allclose(issued, trace["command"], atol=1e-6, rtol=0.)):
                raise ValueError(f"issued command chain mismatch: {episode_path}")
        physical = bool(episode["success"] and not any(
            any(step["physical_flags"].values()) for step in episode["strict_substeps"]))
        no_fallback = bool(physical and episode["fallback_periods"] == 0)
        deadline_clean = bool(episode["success"] and not any(episode["deadline_miss"]))
        recomputed = {"physical_strict": physical,
                      "no_fallback_physical": no_fallback,
                      "deadline_clean": deadline_clean,
                      "strict_mainline": no_fallback and deadline_clean}
        mismatches = [key for key, value in recomputed.items()
                      if bool(episode[key]) != value]
        if mismatches:
            raise ValueError(f"metric recomputation mismatch {mismatches}: {episode_path}")
        episode["geometry"] = Path(episode["path"]).stem
        episodes.append(episode)
        manifests.append({"run": str((episode_path.parent/"run.json").relative_to(args.runs)),
                          "run_sha256": sha(episode_path.parent/"run.json"),
                          "episode": str(episode_path.relative_to(args.runs)),
                          "episode_sha256": sha(episode_path)})
    keys = {(x["geometry"], x["seed"], x["method"]) for x in episodes}
    expected = {(g, s, m) for g in ("dev_reversal", "dev_coupled")
                for s in (32001, 32002, 32003) for m in ("A0", "A4")}
    if keys != expected or len(episodes) != 12:
        raise ValueError(f"expected 12 unique A0/A4 runs; got {len(episodes)}")
    by_key = {(x["geometry"], x["seed"], x["method"]): x for x in episodes}
    pairs = []
    for geometry, seed in sorted({(g, s) for g, s, _ in keys}):
        a, b = by_key[(geometry, seed, "A0")], by_key[(geometry, seed, "A4")]
        pairs.append({"geometry": geometry, "seed": seed,
            "A0_physical_strict": a["physical_strict"],
            "A4_physical_strict": b["physical_strict"],
            "physical_improved": not a["physical_strict"] and b["physical_strict"],
            "physical_worsened": a["physical_strict"] and not b["physical_strict"],
            "A0_deadline_clean": a["deadline_clean"],
            "A4_deadline_clean": b["deadline_clean"],
            "deadline_improved": not a["deadline_clean"] and b["deadline_clean"],
            "deadline_worsened": a["deadline_clean"] and not b["deadline_clean"]})
    summary = {"schema": 1, "scope": "repaired A0/A4 parity; paired unit=geometry x seed",
        "episodes": len(episodes), "paired_units": len(pairs),
        "methods": {m: {"physical_strict": sum(x["physical_strict"] for x in episodes if x["method"] == m),
            "no_fallback_physical": sum(x["no_fallback_physical"] for x in episodes if x["method"] == m),
            "deadline_clean": sum(x["deadline_clean"] for x in episodes if x["method"] == m),
            "fallback_periods": sum(x["fallback_periods"] for x in episodes if x["method"] == m),
            "deadline_miss_periods": sum(sum(x["deadline_miss"]) for x in episodes if x["method"] == m),
            "backup_periods": None} for m in ("A0", "A4")},
        "paired_physical_improved": sum(x["physical_improved"] for x in pairs),
        "paired_physical_worsened": sum(x["physical_worsened"] for x in pairs),
        "paired_deadline_improved": sum(x["deadline_improved"] for x in pairs),
        "paired_deadline_worsened": sum(x["deadline_worsened"] for x in pairs),
        "pairs": pairs, "run_file_manifest": manifests}
    args.out.write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("pairs", "run_file_manifest")}, indent=2))


if __name__ == "__main__":
    main()
