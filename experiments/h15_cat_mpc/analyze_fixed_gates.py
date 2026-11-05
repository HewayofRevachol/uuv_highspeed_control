"""Paired-unit summary for A0--A4 and dev residual forecast errors."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    episodes = [json.loads(p.read_text()) for p in
                sorted(args.runs.glob("*/episode.json"))]
    if any(row.get("command_schema") != "issued_v2" or row.get("schema", 0) < 2
           for row in episodes):
        raise RuntimeError("refusing to mix legacy and issued_v2 episode logs")
    keyed = {(Path(row["path"]).stem, row["seed"], row["method"]): row
             for row in episodes}
    units = sorted({(key[0], key[1]) for key in keyed})
    pair_report = {}
    for method in ("A1", "A2", "A3", "A4"):
        pairs = []
        for geometry, seed in units:
            a, b = keyed[(geometry, seed, "A0")], keyed[(geometry, seed, method)]
            count = min(len(a["trace"]), len(b["trace"]))
            diffs = []
            for i in range(count):
                scale = np.array([1_500_000., np.deg2rad(45.), np.deg2rad(45.)])
                diffs.append(float(np.linalg.norm(
                    (np.asarray(a["trace"][i]["command"])-
                     np.asarray(b["trace"][i]["command"]))/scale)))
            first_violation = lambda row: next(({
                "period": s["period"], "substep": s["substep"],
                "time_s": s["time_s"],
                "types": [kind for kind, flag in s["physical_flags"].items() if flag]}
                for s in row["strict_substeps"]
                if any(s["physical_flags"].values())), None)
            pairs.append({"geometry": geometry, "seed": seed,
                "A0_strict_hist": a["strict_hist"],
                "method_strict_hist": b["strict_hist"],
                "A0_physical_strict": a["physical_strict"],
                "method_physical_strict": b["physical_strict"],
                "A0_strict_mainline": a["strict_mainline"],
                "method_strict_mainline": b["strict_mainline"],
                "improved_physical": not a["physical_strict"] and b["physical_strict"],
                "worsened_physical": a["physical_strict"] and not b["physical_strict"],
                "deadline_improved": not a["deadline_clean"] and b["deadline_clean"],
                "deadline_worsened": a["deadline_clean"] and not b["deadline_clean"],
                "command_diff_nonzero_periods": sum(x > 1e-9 for x in diffs),
                "max_normalized_command_delta": max(diffs, default=0.),
                "first_violations_A0": first_violation(a),
                "first_violations_method": first_violation(b)})
        pair_report[method] = {"pairs": pairs,
            "physical_strict_improved": sum(x["improved_physical"] for x in pairs),
            "physical_strict_worsened": sum(x["worsened_physical"] for x in pairs),
            "deadline_improved": sum(x["deadline_improved"] for x in pairs),
            "deadline_worsened": sum(x["deadline_worsened"] for x in pairs),
            "strict_mainline_A0": sum(x["A0_strict_mainline"] for x in pairs),
            "strict_mainline_method": sum(x["method_strict_mainline"] for x in pairs),
            "pairs_with_command_changes": sum(x["command_diff_nonzero_periods"] > 0
                                                for x in pairs)}
    prediction = {}
    for horizon in ("1", "5", "25"):
        records = [row["residual_forecast_errors"][horizon] for row in episodes
                   if row["residual_forecast_errors"][horizon]["count"]]
        prediction[horizon] = {"episodes": len(records),
            "ar1_mae": np.mean([x["ar1_mae"] for x in records], axis=0).tolist(),
            "persistence_mae": np.mean([x["persistence_mae"] for x in records], axis=0).tolist()}
    result = {"schema": 1, "split": "dev", "episodes": len(episodes),
              "paired_units": len(units), "paired_vs_A0": pair_report,
              "residual_prediction_mae": prediction,
              "note": "paired scenario units; control periods are not treated as independent samples"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
