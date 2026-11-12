#!/usr/bin/env python3
"""Historical-window paired empirical frequency response for the long matrix."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(__import__("os").environ.get("GS12_WEIGHT_OUT", HERE/"output")).resolve()/"long_matrix"
SCALE = np.array([8., 5., 2.])
BANDS = {"B1": np.arange(1, 9), "B4": np.arange(25, 33)}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_runs():
    found = {}
    for p in (OUT/"runs").glob("*.json"):
        rec = json.loads(p.read_text()); tp = OUT/"traces"/(rec["run_id"]+".npz")
        if not tp.exists() or sha(tp) != rec.get("trace_sha256"):
            continue
        key = (rec["task_id"], rec["controller"], rec["a_depth"], rec["a_rudder_rate"])
        found[key] = (rec, np.load(tp))
    return found


def _dc(x, k):
    x = np.asarray(x)
    return 2./len(x) * np.exp(-2j*np.pi*np.arange(len(x))*k/len(x)) @ x


def _at_control(data):
    idx = np.arange(4, len(data["time_substep_s"]), 5)
    return data["time_substep_s"][idx], data["reference_substep"][idx], data["output_substep"][idx]


def _get(records, task_id, method):
    # method is (controller label, a_depth, a_rudder_rate)
    return records.get((task_id, method[0], method[1], method[2]))


def matrix_for(records, band, amplitude, method):
    h_windows = []
    for lo, hi in ((128., 192.), (192., 256.)):
        h = np.zeros((8, 3, 3), dtype=complex)
        for col, axis in enumerate(("east", "depth", "speed")):
            tid = f"{band}_{axis}_a{amplitude:g}"
            entry = _get(records, tid, method)
            zero = _get(records, f"zero_{'C4_candidate' if method[1] != 1. or method[2] != 1. else method[0]}", method)
            if entry is None or zero is None:
                raise KeyError(f"missing pair {tid}, method={method}")
            r, d = entry; zr, zd = zero
            t, ref, out = _at_control(d); tz, refz, outz = _at_control(zd)
            mask = (t > lo+1e-9) & (t <= hi+1e-9)
            zmask = (tz > lo+1e-9) & (tz <= hi+1e-9)
            if np.count_nonzero(mask) != 640 or np.count_nonzero(zmask) != 640:
                raise ValueError("long-protocol window does not contain 640 control samples")
            e = np.column_stack((ref[mask, 1]-out[mask, 1], ref[mask, 2]-out[mask, 2],
                                 ref[mask, 5]-out[mask, 5]))/SCALE
            e0 = np.column_stack((refz[zmask, 1]-outz[zmask, 1], refz[zmask, 2]-outz[zmask, 2],
                                  refz[zmask, 5]-outz[zmask, 5]))/SCALE
            e = e-e0
            input_signal = {"east": (ref[mask, 1]-ref[mask, 0]*0.)/8.,
                            "depth": (ref[mask, 2]-20.)/5.,
                            "speed": (ref[mask, 5]-15.)/2.}[axis]
            # Remove the zero-reference value for the east coordinate (which is zero).
            if axis == "east": input_signal = (ref[mask, 1]-refz[zmask, 1])/8.
            bins = BANDS[band]
            for row, k in enumerate(bins):
                zcoef = _dc(input_signal, int(k))
                if abs(zcoef) <= 1e-10:
                    raise ValueError(f"excitation line k={k} has zero energy")
                h[row, :, col] = _dc(e, int(k))/zcoef
        h_windows.append(h)
    hfinal = (h_windows[0]+h_windows[1])/2.
    period = np.linalg.norm(h_windows[1]-h_windows[0], axis=(1, 2))/np.maximum(
             np.linalg.norm(h_windows[1], axis=(1, 2)), 1e-12)
    sv = np.asarray([np.linalg.svd(m, compute_uv=False) for m in hfinal])
    return {"H_W2": h_windows[0], "H_W3": h_windows[1], "H_final": hfinal,
            "period_relative_change": period, "singular_values": sv}


def enc(a):
    a = np.asarray(a)
    return {"real": a.real.tolist(), "imag": a.imag.tolist()}


def analyze():
    records = read_runs()
    methods = [("C4", 1., 1.), ("C5", 1., 1.)]
    selection = json.loads((HERE/"output"/"selection.json").read_text())
    d, rate = selection["candidates"][0]
    methods.append(("C4", d, rate))
    labels = ["C4", "C5", "C4_weighted"]
    matrices, rows, missing = [], [], []
    for band in ("B1", "B4"):
        for amplitude in (.006, .003):
            for label, method in zip(labels, methods):
                try:
                    result = matrix_for(records, band, amplitude, method)
                except (KeyError, ValueError) as exc:
                    missing.append({"band": band, "amplitude": amplitude, "method": label, "reason": str(exc)})
                    continue
                rec = {"band": band, "amplitude_per_line": amplitude, "method": label,
                       "bins": BANDS[band].tolist(), "omega_radps": (2*np.pi*BANDS[band]/64).tolist(),
                       "H_W2": enc(result["H_W2"]), "H_W3": enc(result["H_W3"]),
                       "H_final": enc(result["H_final"]),
                       "singular_values": result["singular_values"].tolist(),
                       "period_relative_change": result["period_relative_change"].tolist(),
                       "max_period_relative_change": float(np.max(result["period_relative_change"]))}
                matrices.append(rec)
                rows.append({"band": band, "amplitude": amplitude, "method": label,
                             "mean_sigma1": float(np.mean(result["singular_values"][:, 0])),
                             "peak_sigma1": float(np.max(result["singular_values"][:, 0])),
                             "max_W2_W3_relative_change": rec["max_period_relative_change"],
                             "period_quality_le_2pct": bool(rec["max_period_relative_change"] <= .02)})
    amp_diffs = []
    for band in ("B1", "B4"):
        for method in labels:
            full = next((m for m in matrices if m["band"] == band and m["method"] == method and m["amplitude_per_line"] == .006), None)
            half = next((m for m in matrices if m["band"] == band and m["method"] == method and m["amplitude_per_line"] == .003), None)
            if full and half:
                decode = lambda x: np.asarray(x["real"])+1j*np.asarray(x["imag"])
                hf, hh = decode(full["H_final"]), decode(half["H_final"])
                amp_diffs.append({"band": band, "method": method,
                    "relative_matrix_difference_half_vs_full": float(np.linalg.norm(hh-hf)/max(np.linalg.norm(hf), 1e-12))})
    (OUT/"long_matrices.json").write_text(json.dumps({"matrices": matrices, "missing": missing,
           "half_vs_full": amp_diffs, "protocol_sha256": sha(HERE/"output"/"long_protocol.json")}, indent=2)+"\n")
    if rows:
        with (HERE/"output"/"long_matrix_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {"runs": len(records), "matrix_groups": len(matrices), "missing_groups": missing,
               "half_vs_full": amp_diffs, "matrix_metrics": rows}
    (OUT/"long_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"runs": len(records), "matrix_groups": len(matrices), "missing_groups": len(missing)}))


if __name__ == "__main__": analyze()
