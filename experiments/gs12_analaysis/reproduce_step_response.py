"""Recompute the repository's fixed-mismatch 12/15/18 m/s rudder step probe.

Example:
    python reproduce_step_response.py --repo ../uuv_highspeed_control

The repository must be checked out at the commit recorded in this report.
The experiment uses its original evaluate() function without modifying it.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=directory.parent / "uuv_highspeed_control",
                        help="repository checkout; default is the adjacent uuv_highspeed_control directory")
    parser.add_argument("--out", type=Path, default=directory / "step_response.json",
                        help="output JSON; default is step_response.json beside this script")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    if not (repo / "aav_mpc/h15_step_response.py").is_file():
        parser.error(f"repository function not found under {repo}; pass --repo explicitly")
    sys.path.insert(0, str(repo))
    from aav_mpc.h15_step_response import evaluate
    from simulator.model import DT

    report = {"plant": "fixed_mismatch", "integration_period_s": DT,
              "results": [evaluate(speed, fin_deg=5., duration_s=30.)
                          for speed in (12., 15., 18.)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
