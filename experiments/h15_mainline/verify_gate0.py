"""Audit frozen H15 Gate0 inputs and logging contracts before Gate1."""

import argparse
import hashlib
import json
from pathlib import Path
import stat

from experiments.h15_mainline.cycle_log import validate as validate_cycle
from experiments.h15_mainline.cycle_writer import timing_summary
from experiments.h15_mainline.run_record import (RESULT_FIELDS, TIMING_STAGES,
                                                  TOP_LEVEL, validate as validate_run,
                                                  write_once)


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--schema-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    task = ROOT / "experiments/h15_mainline/task_contract.json"
    config = ROOT / "aav_mpc/configs/h15.json"
    if freeze["task_contract_sha256"] != sha(task):
        raise RuntimeError("task contract hash differs from freeze")
    if freeze["task_config_sha256"] != sha(config):
        raise RuntimeError("H15 task config hash differs from freeze")
    contract = json.loads(task.read_text())
    for field in ("strict_hist", "strict_mainline", "completed_with_backup"):
        if field not in contract:
            raise RuntimeError(f"missing strict contract field {field}")
    manifests = {}
    for split, entry in freeze["manifests"].items():
        path = ROOT / entry["path"]
        if sha(path) != entry["sha256"]:
            raise RuntimeError(f"changed {split} manifest")
        manifest = json.loads(path.read_text())
        for case in manifest["cases"]:
            file = ROOT / case["path"]
            if sha(file) != case["path_sha256"]:
                raise RuntimeError(f"changed {split} path {file}")
            if split == "test" and stat.S_IMODE(file.stat().st_mode) & 0o222:
                raise RuntimeError(f"test path is writable: {file}")
        if split == "test" and stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise RuntimeError("test manifest is writable")
        manifests[split] = {"sha256": entry["sha256"],
                            "cases": len(manifest["cases"]),
                            "seeds": len(manifest["seeds"])}
    if freeze["candidate_limit_k"] != len(freeze["candidate_grid_mps"]):
        raise RuntimeError("frozen candidate count differs from grid")
    if freeze["screen_budget_ms"] <= 0:
        raise RuntimeError("invalid frozen screening budget")
    normalization = json.loads(args.normalization.read_text())
    if (normalization["split"] != "train" or
            normalization["train_manifest_sha256"] != freeze["manifests"]["train"]["sha256"]):
        raise RuntimeError("normalization not fitted on frozen train")
    if len(normalization["trace_sha256"]) != 12:
        raise RuntimeError("incomplete train normalization inputs")
    for name, digest in normalization["trace_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"changed normalization input {name}")
    schema_run = json.loads((args.schema_run / "run.json").read_text())
    validate_run(schema_run)
    if schema_run["method"] != "h15_reference_chain_smoke_profiled":
        raise RuntimeError("Gate0 schema probe is not diagnostic smoke")
    for source in ("run_record.py", "cycle_log.py", "cycle_writer.py"):
        if schema_run["config_sha256"][source] != sha(ROOT / f"experiments/h15_mainline/{source}"):
            raise RuntimeError(f"log schema source changed: {source}")
    if schema_run["config_sha256"]["quick_run_manifest.json"] != sha(args.freeze):
        raise RuntimeError("schema smoke uses another freeze")
    if schema_run["results"]["strict_mainline_completed"] is not None:
        raise RuntimeError("diagnostic smoke claimed formal mainline success")
    cycles = []
    with (args.schema_run / "cycle_log.jsonl").open() as stream:
        for line in stream:
            cycles.append(validate_cycle(json.loads(line)))
    if len(cycles) != 1233 or len({row["episode_id"] for row in cycles}) != 6:
        raise RuntimeError("incomplete smoke cycle schema validation")
    if timing_summary(cycles) != schema_run["timing_ms"]:
        raise RuntimeError("run timing differs from cycle aggregation")
    if any(row["candidate_empty"] is not None or row["raw_rl_intent_mps"] is not None
           for row in cycles):
        raise RuntimeError("reference smoke contains fabricated method events")
    report = {"schema": 1, "gate": "Gate0", "passed": True,
              "task_config_sha256": sha(config), "task_contract_sha256": sha(task),
              "manifests": manifests, "candidate_limit_k": freeze["candidate_limit_k"],
              "screen_budget_ms": freeze["screen_budget_ms"],
              "normalization_sha256": sha(args.normalization),
              "diagnostic_cycles_validated": len(cycles),
              "schema_run_sha256": sha(args.schema_run / "run.json"),
              "meaning": "frozen configuration, manifests and logging schema; formal method values remain unmeasured"}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    results = dict.fromkeys(RESULT_FIELDS)
    results.update(episodes=1, first_violations={})
    record = dict(zip(TOP_LEVEL, (1, "h15_gate0_audit",
                                  {"quick_run_manifest.json": sha(args.freeze),
                                   "task_contract.json": sha(task),
                                   "run_record.py": sha(ROOT / "experiments/h15_mainline/run_record.py"),
                                   "cycle_log.py": sha(ROOT / "experiments/h15_mainline/cycle_log.py"),
                                   "cycle_writer.py": sha(ROOT / "experiments/h15_mainline/cycle_writer.py")},
                                  {f"{split}_manifest.json": value["sha256"]
                                   for split, value in manifests.items()} |
                                  {"normalization.json": sha(args.normalization)},
                                  [31001, 31002, 31003],
                                  dict.fromkeys(TIMING_STAGES), results)))
    write_once(args.out / "run.json", record)
    (args.out / "console.log").write_text(
        f"Gate0 frozen inputs and log schema validated; {len(cycles)} diagnostic cycles\n")
    print(json.dumps({"passed": True, "cycles": len(cycles),
                      "manifests": manifests}, indent=2))


if __name__ == "__main__":
    main()
