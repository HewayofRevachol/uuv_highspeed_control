"""Build the frozen dev SVD direction from valid single-frequency axis runs."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from experiments.gs12_gate.data_io import immutable_json
from experiments.gs12_gate.episode import run_episode
from experiments.gs12_gate.frequency import (
    FREQUENCY_BINS, FREQUENCIES, PILOT_LOW, PERIOD, _analysis,
    _complex_json, _case,
)
from experiments.gs12_gate.plant_harness import make_case
from experiments.gs12_gate.run import _artifacts, _load_run


def build(run):
    _load_run(run)
    artifacts = _artifacts(run)
    root = Path(run) / "frequency" / "dev_direction"
    zero = root / "pilot_zero"
    baseline = make_case("dev_direction_pilot_zero", "dev_direction", 15,
                         "second_all", "frequency_direction", 81000,
                         duration_s=144.)
    baseline.update(amplitude=0., omega=float(PILOT_LOW*2*np.pi/PERIOD),
                    complex_direction=_complex_json([1., 0., 0.]))
    baseline["plant"]["fin"] = .9
    run_episode(baseline, "hybrid_zero", zero, artifacts=artifacts)

    columns = []
    omega = FREQUENCIES[FREQUENCY_BINS.index(PILOT_LOW)]
    for axis in range(3):
        target = root / f"pilot_single_axis_{axis}"
        case = make_case(f"dev_direction_pilot_single_axis_{axis}",
                         "dev_direction", 15, "second_all",
                         "frequency_direction", 81001+axis, duration_s=144.)
        case.update(amplitude=.006, omega=float(omega),
                    complex_direction=_complex_json(np.eye(3, dtype=complex)[axis]))
        case["plant"]["fin"] = .9
        result = run_episode(case, "hybrid_zero", target, artifacts=artifacts)
        if not result["complete"]:
            raise RuntimeError(f"development axis case failed: {case['case_id']}")
        metric = _analysis(target, zero, omega,
                           np.eye(3, dtype=complex)[axis], .006, 16.,
                           excited_bins=(PILOT_LOW,))
        if not metric["valid"]:
            raise RuntimeError(f"invalid unsaturated single-frequency FRF: axis={axis}, "
                               f"amplitude_ratio={metric['amplitude_ratio']}")
        columns.append(metric["error_complex"] / metric["input_complex"][axis])

    matrix = np.column_stack(columns)
    _, singular, vh = np.linalg.svd(matrix)
    vector = vh.conj().T[:, 0]
    vector *= np.exp(-1j * np.angle(vector[np.argmax(np.abs(vector))]))
    source = {
        "split": "dev", "frequency_bin": PILOT_LOW,
        "omega_rad_s": float(omega),
        "right_singular_vector": _complex_json(vector),
        "singular_values": singular.tolist(),
        "H_real": matrix.real.tolist(), "H_imag": matrix.imag.tolist(),
        "source": "single_frequency_dev_hybrid_zero_axis_runs",
        "case_ids": ["dev_direction_zero"] +
            [f"dev_direction_pilot_single_axis_{j}" for j in range(3)],
    }
    immutable_json(root / "direction_source.json", source)
    return source


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.run))
