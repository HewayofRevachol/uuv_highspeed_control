"""Audit frozen reference-chain smoke traces without claiming mainline results."""

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

from experiments.h15_mainline.strict_contract import evaluate


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit(summary):
    trace = ROOT / summary["trace"]
    periods = read_csv(trace)
    substeps = read_csv(trace.with_name(trace.stem + "_substeps.csv"))
    assert len(periods) == summary["steps"]
    assert len(substeps) == 5 * len(periods)
    assert [int(row["period"]) for row in substeps] == [i for i in range(len(periods)) for _ in range(5)]
    assert [int(row["substep"]) for row in substeps] == list(range(1, 6)) * len(periods)
    rates = [float(row["nu_ref_mps"]) for row in periods]
    assert all(math.isfinite(rate) and 13. <= rate <= 17. for rate in rates)
    assert all(abs(right - left) <= .0500001 for left, right in zip([15.] + rates[:-1], rates))
    for i, row in enumerate(periods):
        assert json.loads(row["queue_before_json"])[0] == (
            json.loads(periods[i-1]["command_json"]) if i else [0., 0., 0.])
        last = substeps[5*i+4]
        actual = [float(row["thrust_actual_n"]),
                  math.radians(float(row["elevator_actual_deg"])),
                  math.radians(float(row["rudder_actual_deg"]))]
        assert all(abs(a-b) < 1e-8 for a, b in zip(actual, json.loads(last["actuator_json"])))
    for row in periods + substeps:
        fields = ("lateral_error_m", "depth_error_m", "speed_error_mps", "pitch_deg")
        assert all(math.isfinite(float(row[field])) for field in fields)
    result = evaluate(
        completed=bool(summary["simulator_success"]), period_samples=periods,
        substep_samples=substeps,
        mpc_fallback=summary["fallback_count"] > 0, candidate_empty=None,
        backup_used=None, deadline_missed=None, fallback_defined=None)
    assert result["strict_hist"] == summary["strict_hist"]
    assert (result["first_substep_violation"] is None) == bool(summary["strict_substep_success"])
    assert result["strict_mainline"] is None
    return {"trace": summary["trace"], "steps": len(periods),
            "substeps": len(substeps), "strict_hist": result["strict_hist"],
            "first_substep_violation": result["first_substep_violation"],
            "strict_mainline": None,
            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summaries = json.loads((args.smoke / "summary.json").read_text())
    assert len(summaries) == 6
    results = [audit(summary) for summary in summaries]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run = {"schema": 1, "method": "h15_reference_chain_audit",
           "input_summary": str((args.smoke / "summary.json").resolve().relative_to(ROOT)),
           "input_summary_sha256": hashlib.sha256((args.smoke / "summary.json").read_bytes()).hexdigest(),
           "sources_sha256": {str(Path(__file__).resolve().relative_to(ROOT)):
                              hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "experiments/h15_mainline/strict_contract.py": hashlib.sha256(
                                  (ROOT / "experiments/h15_mainline/strict_contract.py").read_bytes()).hexdigest()},
           "python_executable": "/bin/python3", "python": platform.python_version(),
           "output": str(args.out.resolve().relative_to(ROOT))}
    (args.out.parent / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    args.out.write_text(json.dumps({"schema": 1, "scope": "reference_chain_smoke_only",
                                    "episodes": results}, indent=2) + "\n")
    (args.out.parent / "console.log").write_text(
        f"audited {len(results)} reference-chain traces; formal mainline status unavailable\n")
    print(f"audited {len(results)} reference-chain traces; formal mainline status unavailable")


if __name__ == "__main__":
    main()
