"""Empirical GS-12 directional and axis-frequency probes.

Dynamic-gate results are empirical responses at the declared stimulus and
clock phase. A matrix assembled from separate axis runs is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.gs12_gate.data_io import file_hash, immutable_json, read_json, write_npz
from experiments.gs12_gate.episode import run_episode
from experiments.gs12_gate.features import GateActorCritic
from experiments.gs12_gate.plant_harness import make_case, capacity_limited
from experiments.gs12_gate.run import _artifacts, _load_run
from simulator.model import rotation_ned

PERIOD = 64.
FREQUENCY_BINS = (2, 4, 6, 8, 12, 16, 24)
FREQUENCIES = np.asarray(FREQUENCY_BINS)*2*np.pi/PERIOD
PILOT_LOW = 6
PILOT_SPEED = 24


def _complex_json(vector):
    v = np.asarray(vector, complex)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("direction must be a finite complex three-vector")
    v = v/np.linalg.norm(v)
    return [[float(z.real), float(z.imag)] for z in v]


def _case(identifier, bin_number, direction, amplitude, duration, phase=0.):
    case = make_case(identifier, "frequency", 15, "nominal",
                     "frequency_direction", 81000+bin_number,
                     duration_s=duration)
    # Match the GS-12 FRF plant: fin loss plus actuator order, with nominal
    # time constants and delay. The general combination family adds both.
    case["family"] = "combination"
    case["plant"].update(fin=.9, order="second_all")
    case.update(amplitude=float(amplitude), omega=float(bin_number*2*np.pi/PERIOD),
                phase=float(phase), complex_direction=_complex_json(direction))
    return case


def _policy(checkpoint, run):
    if checkpoint is None: return None
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for field, name in (("normalization_hash", "normalization.json"),
                        ("residual_hash", "residual_scale.json"),
                        ("schema_hash", "obs_schema.json")):
        if saved[field] != file_hash(Path(run)/name):
            raise ValueError(f"checkpoint {field} mismatch")
    policy = GateActorCritic(saved.get("policy_variant", "standard"))
    policy.load_state_dict(saved["policy"]); policy.eval()
    return policy


def _checkpoint_label(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    variant = saved.get("policy_variant", "standard")
    return f"ppo_{saved['seed']}" if variant == "standard" else f"{variant}_{saved['seed']}"


def _load_error(run_dir, start_s, duration_s=64.):
    with np.load(Path(run_dir)/"control.npz") as data:
        t = data["t"]
        select = (t >= start_s-1e-9) & (t < start_s+duration_s-1e-9)
        state, reference = data["state"][select], data["reference"][select]
        selected_t = t[select]
        if len(selected_t) != int(round(duration_s/.1)):
            raise ValueError("incomplete frequency analysis window")
        velocity_n = np.array([(rotation_ned(x[3], x[4])@x[5:8])[0] for x in state])
        error = np.column_stack(((state[:, 1]-reference[:, 1])/8,
                                 (state[:, 2]-reference[:, 2])/5,
                                 (velocity_n-reference[:, 5])/2))
        input_signal = reference[:, [1, 2, 5]]/[8., 5., 2.]
        return selected_t, error, input_signal, bool(np.all(data["qp_ok"][select]) and
                                                     not np.any(data["saturation_fraction"][select] > 0))


def _spectrum(t, value, omega):
    return 2*np.mean(value*np.exp(-1j*omega*t)[:, None], axis=0)


def _analysis(stimulus_dir, baseline_dir, omega, direction, amplitude,
              start_s, duration_s=64., excited_bins=None):
    t, error, input_signal, qp_ok = _load_error(stimulus_dir, start_s, duration_s)
    if baseline_dir is None:
        baseline = np.zeros_like(error)
        baseline_input = np.zeros_like(input_signal)
        base_qp = True
    else:
        t0, baseline, baseline_input, base_qp = _load_error(baseline_dir, start_s, duration_s)
        if not np.allclose(t, t0): raise ValueError("stimulus/baseline time mismatch")
    difference = error-baseline
    output = _spectrum(t, difference, omega)
    input_complex = _spectrum(t, input_signal-baseline_input, omega)
    fft = np.fft.rfft(difference, axis=0)
    stimulated_bin = int(round(omega*duration_s/(2*np.pi)))
    for index in (excited_bins if excited_bins is not None else [stimulated_bin]):
        fft[index] = 0
    spectrum = np.sum(np.abs(fft)**2, axis=1)/len(t)**2
    # rfft omits the negative frequencies; retain Parseval mean-square units.
    spectrum[1:-1 if len(t) % 2 == 0 else None] *= 2
    amplitude_ratio = float(np.linalg.norm(input_complex)/max(amplitude, 1e-12))
    return {"input_complex": input_complex, "error_complex": output,
            "direction_gain": float(np.linalg.norm(output)/max(np.linalg.norm(input_complex), 1e-12)),
            "total_error_rms": float(np.sqrt(np.mean(np.sum(error*error, axis=1)))),
            "nonexcited_energy": float(np.sum(spectrum)),
            "nonexcited_spectrum": spectrum,
            "spectrum_omega": np.fft.rfftfreq(len(t), d=.1)*2*np.pi,
            "amplitude_ratio": amplitude_ratio,
            "valid": bool(qp_ok and base_qp and abs(amplitude_ratio-1.) <= .05)}


def _save_metrics(path, metrics, metadata):
    arrays = {"input_real": np.real(metrics["input_complex"]),
              "input_imag": np.imag(metrics["input_complex"]),
              "error_real": np.real(metrics["error_complex"]),
              "error_imag": np.imag(metrics["error_complex"]),
              "direction_gain": np.asarray(metrics["direction_gain"]),
              "total_error_rms": np.asarray(metrics["total_error_rms"]),
              "nonexcited_energy": np.asarray(metrics["nonexcited_energy"]),
              "amplitude_ratio": np.asarray(metrics["amplitude_ratio"]),
              "nonexcited_spectrum": metrics["nonexcited_spectrum"],
              "spectrum_omega": metrics["spectrum_omega"],
              "valid": np.asarray(metrics["valid"])}
    write_npz(path.with_suffix(".npz"), **arrays)
    immutable_json(path.with_suffix(".json"), metadata)


def pilot(run, direction_source, checkpoint=None, methods=None):
    _load_run(run)
    artifacts = _artifacts(run)
    source = read_json(direction_source)
    if source.get("split") not in ("dev", "dev_screen", "dev_quick"):
        raise ValueError("pilot direction must be frozen from development data")
    if source.get("frequency_bin") != PILOT_LOW:
        raise ValueError("direction source must specify 64 s bin 6")
    low_direction = np.asarray(source["right_singular_vector"], float)
    low = low_direction[:, 0]+1j*low_direction[:, 1]
    speed = np.array([0., 0., 1.], complex)
    selection = read_json(Path(run)/"selected_baselines.json")
    names = methods or ["fixed_best", "rule_best", "ppo"]
    policy = _policy(checkpoint, run)
    records = []
    for name in names:
        method = selection.get(name, name)
        if name == "ppo": method = "ppo_mean"
        for label, bin_number, direction in (("low", PILOT_LOW, low),
                                             ("speed", PILOT_SPEED, speed)):
            case = _case(f"pilot_{name}_{label}", bin_number, direction, .006, 80.)
            target = Path(run)/"frequency"/"pilot"/name/label
            result = run_episode(case, method, target, artifacts=artifacts, policy=policy)
            if not result["complete"]:
                metrics = {"input_complex": np.full(3, np.nan+1j*np.nan),
                           "error_complex": np.full(3, np.nan+1j*np.nan),
                           "direction_gain": np.nan, "total_error_rms": np.nan,
                           "nonexcited_energy": np.nan, "amplitude_ratio": np.nan,
                           "nonexcited_spectrum": np.full(321, np.nan),
                           "spectrum_omega": np.fft.rfftfreq(640, d=.1)*2*np.pi,
                           "valid": False}
            else:
                metrics = _analysis(target, None, case["omega"],
                                    direction/np.linalg.norm(direction), .006, 16.)
            metadata = {"method": name, "frequency_bin": bin_number,
                        "omega_rad_s": case["omega"], "analysis_window_s": [16., 80.],
                        "amplitude": .006, "direction_source_hash": file_hash(direction_source),
                        "empirical_dynamic_response": True, "valid": metrics["valid"]}
            _save_metrics(target/"frequency", metrics, metadata)
            records.append({**metadata,
                            "direction_gain": metrics["direction_gain"] if metrics["valid"] else None,
                            "nonexcited_energy": metrics["nonexcited_energy"] if metrics["valid"] else None})
    immutable_json(Path(run)/"frequency"/"pilot"/"summary.json", records)
    return records


def freeze_direction(run):
    """Reconstruct the missing development FRF direction from this codebase."""
    _load_run(run)
    artifacts = _artifacts(run)
    baseline = _case("dev_direction_zero", PILOT_LOW, np.array([1., 0., 0.]), 0., 144.)
    baseline["split"] = "dev_direction"
    root = Path(run)/"frequency"/"dev_direction"
    zero = root/"zero"
    run_episode(baseline, "hybrid_zero", zero, artifacts=artifacts)
    columns = []
    for axis in range(3):
        direction = np.eye(3, dtype=complex)[axis]
        case = _case(f"dev_direction_axis_{axis}", PILOT_LOW, direction, .006, 144.)
        case["split"] = "dev_direction"
        case["multi_sine_bins"] = list(FREQUENCY_BINS)
        target = root/f"axis_{axis}"
        result = run_episode(case, "hybrid_zero", target, artifacts=artifacts)
        if not result["complete"]:
            raise RuntimeError(f"development FRF case failed: {case['case_id']}")
        metrics = _analysis(target, zero, FREQUENCIES[FREQUENCY_BINS.index(PILOT_LOW)],
                            direction, .006, 16., excited_bins=FREQUENCY_BINS)
        if not metrics["valid"]:
            raise RuntimeError("development direction FRF has saturation, QP failure or amplitude mismatch")
        columns.append(metrics["error_complex"]/metrics["input_complex"][axis])
    matrix = np.column_stack(columns)
    u, s, vh = np.linalg.svd(matrix)
    vector = vh.conj().T[:, 0]
    vector *= np.exp(-1j*np.angle(vector[np.argmax(np.abs(vector))]))
    source = {"split": "dev", "frequency_bin": PILOT_LOW,
              "omega_rad_s": float(FREQUENCIES[FREQUENCY_BINS.index(PILOT_LOW)]),
              "right_singular_vector": _complex_json(vector),
              "singular_values": s.tolist(),
              "H_real": matrix.real.tolist(), "H_imag": matrix.imag.tolist(),
              "source": "reconstructed_dev_hybrid_zero_multisine",
              "case_ids": ["dev_direction_zero"]+[f"dev_direction_axis_{j}" for j in range(3)]}
    immutable_json(root/"direction_source.json", source)
    return source


def confirm(run, checkpoints, methods=None):
    _load_run(run)
    artifacts = _artifacts(run)
    selection = read_json(Path(run)/"selected_baselines.json")
    names = methods or ["fixed_best", "rule_best"] + [_checkpoint_label(path)
                                                     for path in checkpoints]
    if len(names) != len(checkpoints)+2:
        raise ValueError("confirm expects two baselines and one name per checkpoint")
    records = []
    for index, name in enumerate(names):
        method = selection.get(name, name)
        policy = None
        if index >= 2:
            method = "ppo_mean"; policy = _policy(checkpoints[index-2], run)
        zero_case = _case(f"confirm_{name}_zero", 2, np.array([1., 0., 0.]), 0., 144.)
        zero_dir = Path(run)/"frequency"/"confirm"/name/"zero"
        run_episode(zero_case, method, zero_dir, artifacts=artifacts, policy=policy)
        amplitude_results = []
        for amplitude in (.006, .003):
            axis_outputs = []
            for axis in range(3):
                direction = np.eye(3, dtype=complex)[axis]
                case = _case(f"confirm_{name}_{amplitude}_axis_{axis}", 2,
                             direction, amplitude, 144.)
                case["multi_sine_bins"] = list(FREQUENCY_BINS)
                target = Path(run)/"frequency"/"confirm"/name/f"axis_{axis}_{amplitude}"
                result = run_episode(case, method, target, artifacts=artifacts, policy=policy)
                if not result["complete"]:
                    axis_outputs.append(None); continue
                outputs = []
                for omega in FREQUENCIES:
                    outputs.append((
                        _analysis(target, zero_dir, omega, direction,
                                  amplitude, 16., excited_bins=FREQUENCY_BINS),
                        _analysis(target, zero_dir, omega, direction,
                                  amplitude, 80., excited_bins=FREQUENCY_BINS)))
                axis_outputs.append(outputs)
            amplitude_results.append(axis_outputs)
        # Separate-axis adaptive responses cannot be identified as one LTI MIMO FRF.
        if (all(all(v is not None for v in group) for group in amplitude_results) and
                all(metrics["valid"] for group in amplitude_results
                    for axis in group for pair in axis for metrics in pair)):
            matrices, second_matrices = [], []
            for amplitude, group in zip((.006, .003), amplitude_results):
                matrices.append(np.stack([np.column_stack(
                    [group[j][i][0]["error_complex"]/
                     group[j][i][0]["input_complex"][j] for j in range(3)])
                    for i in range(len(FREQUENCIES))]))
                second_matrices.append(np.stack([np.column_stack(
                    [group[j][i][1]["error_complex"]/
                     group[j][i][1]["input_complex"][j] for j in range(3)])
                    for i in range(len(FREQUENCIES))]))
            h, h_half = matrices
            h_second, h_half_second = second_matrices
            u, singular, vh = np.linalg.svd(h)
            all_valid = all(metrics["valid"] for group in amplitude_results
                            for axis in group for pair in axis for metrics in pair)
            write_npz(Path(run)/"frequency"/"confirm"/name/"frequency.npz",
                      omega=FREQUENCIES, H_real=h.real, H_imag=h.imag,
                      H_second_real=h_second.real, H_second_imag=h_second.imag,
                      H_half_real=h_half.real, H_half_imag=h_half.imag,
                      H_half_second_real=h_half_second.real,
                      H_half_second_imag=h_half_second.imag,
                      U_real=u.real, U_imag=u.imag,
                      V_real=vh.conj().transpose(0, 2, 1).real,
                      V_imag=vh.conj().transpose(0, 2, 1).imag,
                      singular_values=singular, valid=np.asarray(all_valid),
                      nonexcited_energy=np.asarray([[amplitude_results[0][j][i][0]["nonexcited_energy"]
                                                     for j in range(3)] for i in range(len(FREQUENCIES))]),
                      nonexcited_spectrum=np.stack(
                          [amplitude_results[0][j][0][0]["nonexcited_spectrum"] for j in range(3)]),
                      spectrum_omega=amplitude_results[0][0][0][0]["spectrum_omega"],
                      relative_window_difference=np.linalg.norm(h-h_second, axis=(1, 2))/
                          np.maximum(np.linalg.norm(h, axis=(1, 2)), 1e-12),
                      relative_half_difference=np.linalg.norm(h-h_half, axis=(1, 2))/
                          np.maximum(np.linalg.norm(h, axis=(1, 2)), 1e-12))
            valid = all_valid
        else:
            valid = False
        records.append({"method": name, "valid": valid,
                        "description": "separate-axis empirical response; singular values diagnostic only"})
    immutable_json(Path(run)/"frequency"/"confirm"/"summary.json", records)
    return records


def locked(run, checkpoints):
    """Adapt for 16 s, freeze gate/theta/residual and reuse one snapshot per seed."""
    _load_run(run)
    artifacts = _artifacts(run)
    records = []
    for checkpoint in checkpoints:
        policy = _policy(checkpoint, run)
        name = _checkpoint_label(checkpoint)
        root = Path(run)/"frequency"/"locked"/name
        calibration = _case(f"locked_{name}_calibration", 6,
                            np.array([0., 1., 0.]), .006, 16.)
        calibration["split"] = "dev_locked"
        pre = root/"calibration"
        snapshot_path = pre/"snapshot.pt"
        pre_result = run_episode(calibration, "ppo_mean", pre, artifacts=artifacts,
                                 policy=policy, snapshot_out=snapshot_path)
        if not pre_result["complete"]:
            raise RuntimeError(f"locked calibration failed for {name}")
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        origin = float(snapshot["plant_state"][0])
        axes = [np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                np.array([0., 0., 1.])]
        baseline = _case(f"locked_{name}_zero", 2, axes[0], 0., 144.)
        baseline.update(split="dev_locked", frequency_origin_time_s=16.,
                        frequency_origin_north_m=origin)
        zero = root/"zero"
        run_episode(baseline, "ppo_mean", zero, artifacts=artifacts,
                    policy=policy, snapshot_path=snapshot_path, freeze_gate=True)
        columns = []
        for axis, direction in enumerate(axes):
            case = _case(f"locked_{name}_axis_{axis}", 2, direction, .006, 144.)
            case.update(split="dev_locked", multi_sine_bins=list(FREQUENCY_BINS),
                        frequency_origin_time_s=16.,
                        frequency_origin_north_m=origin)
            target = root/f"axis_{axis}"
            result = run_episode(case, "ppo_mean", target, artifacts=artifacts,
                                 policy=policy, snapshot_path=snapshot_path,
                                 freeze_gate=True)
            if not result["complete"]:
                columns.append(None); continue
            column = []
            for omega in FREQUENCIES:
                first = _analysis(target, zero, omega, direction, .006, 32.,
                                  excited_bins=FREQUENCY_BINS)
                second = _analysis(target, zero, omega, direction, .006, 96.,
                                   excited_bins=FREQUENCY_BINS)
                column.append((first, second))
            columns.append(column)
        valid = (all(x is not None for x in columns) and
                 all(metrics["valid"] for column in columns if column is not None
                     for pair in column for metrics in pair))
        if valid:
            h_first = np.stack([np.column_stack(
                [columns[j][i][0]["error_complex"]/
                 columns[j][i][0]["input_complex"][j]
                 for j in range(3)])
                for i in range(len(FREQUENCIES))])
            h_second = np.stack([np.column_stack(
                [columns[j][i][1]["error_complex"]/
                 columns[j][i][1]["input_complex"][j]
                 for j in range(3)])
                for i in range(len(FREQUENCIES))])
            u, singular, vh = np.linalg.svd(h_first)
            valid = all(metrics["valid"] for column in columns
                        for pair in column for metrics in pair)
            write_npz(root/"frequency.npz", omega=FREQUENCIES,
                      H_real=h_first.real, H_imag=h_first.imag,
                      H_second_real=h_second.real, H_second_imag=h_second.imag,
                      U_real=u.real, U_imag=u.imag,
                      V_real=vh.conj().transpose(0, 2, 1).real,
                      V_imag=vh.conj().transpose(0, 2, 1).imag,
                      singular_values=singular, valid=np.asarray(valid))
        records.append({"seed": name, "checkpoint_hash": file_hash(checkpoint),
                        "snapshot_hash": file_hash(snapshot_path), "valid": valid,
                        "fixed_theta": snapshot["theta_used"].tolist(),
                        "fixed_residual": snapshot["d_used"].tolist(),
                        "fixed_alpha": snapshot["alpha"].tolist()})
    immutable_json(Path(run)/"frequency"/"locked"/"summary.json", records)
    return records


def generalize(run, checkpoints, bins=(3, 5, 10, 20), gate_phases=(1, 2, 3, 4)):
    """New phase, untrained integer bins and 18 m/s direction-subspace replay."""
    _load_run(run)
    artifacts = _artifacts(run)
    selection = read_json(Path(run)/"selected_baselines.json")
    names = ["fixed_best", "rule_best"]+[_checkpoint_label(path)
                                         for path in checkpoints]
    policies = {name: _policy(path, run) for name, path in zip(names[2:], checkpoints)}
    rng = np.random.default_rng(88001)
    directions = {"east": np.array([1., 0., 0.], complex),
                  "depth": np.array([0., 1., 0.], complex),
                  "speed": np.array([0., 0., 1.], complex)}
    random_vector = rng.normal(size=3)+1j*rng.normal(size=3)
    directions["complex_random"] = random_vector/np.linalg.norm(random_vector)
    records = []
    for name in names:
        method = "ppo_mean" if name in policies else selection[name]
        policy = policies.get(name)
        for bin_number in bins:
            if bin_number in FREQUENCY_BINS:
                raise ValueError("generalization bins must be untrained")
            for phase_steps in gate_phases:
                for direction_name, direction in directions.items():
                    case = _case(f"generalize_{name}_{bin_number}_{phase_steps}_{direction_name}",
                                 bin_number, direction, .006, 80., phase=.37)
                    case["split"] = "dev_generalization"
                    case["speed"] = 18.
                    case["capacity_limited"] = capacity_limited(case)
                    target = (Path(run)/"frequency"/"generalization"/name/
                              f"bin_{bin_number}"/f"phase_{phase_steps}"/direction_name)
                    summary = run_episode(case, method, target, artifacts=artifacts,
                                          policy=policy, gate_phase_steps=phase_steps)
                    metric = (_analysis(target, None, case["omega"],
                                        direction/np.linalg.norm(direction), .006, 16.)
                              if summary["complete"] else None)
                    records.append({"method": name, "case_id": case["case_id"],
                                    "frequency_bin": bin_number,
                                    "gate_phase_steps": phase_steps,
                                    "reference_phase_rad": .37,
                                    "direction": direction_name,
                                    "valid": bool(metric is not None and metric["valid"]),
                                    "direction_gain": (metric["direction_gain"]
                                                       if metric is not None and metric["valid"] else None),
                                    "nonexcited_energy": (metric["nonexcited_energy"]
                                                          if metric is not None and metric["valid"] else None)})
    immutable_json(Path(run)/"frequency"/"generalization"/"summary.json", records)
    return {"episodes": len(records), "valid": sum(row["valid"] for row in records)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=("freeze-direction", "pilot", "confirm",
                                            "locked", "generalize"), required=True)
    parser.add_argument("--direction-source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoints")
    parser.add_argument("--methods")
    parser.add_argument("--bins", default="3,5,10,20")
    parser.add_argument("--gate-phases", default="1,2,3,4")
    args = parser.parse_args()
    if args.mode == "freeze-direction":
        result = freeze_direction(args.run)
    elif args.mode == "pilot":
        if args.direction_source is None: raise ValueError("pilot requires --direction-source")
        result = pilot(args.run, args.direction_source, args.checkpoint,
                       args.methods.split(",") if args.methods else None)
    elif args.mode == "confirm":
        paths = [Path(p) for p in (args.checkpoints or "").split(",") if p]
        result = confirm(args.run, paths,
                         args.methods.split(",") if args.methods else None)
    elif args.mode == "locked":
        paths = [Path(p) for p in (args.checkpoints or "").split(",") if p]
        if not paths: raise ValueError("locked mode requires --checkpoints")
        result = locked(args.run, paths)
    else:
        paths = [Path(p) for p in (args.checkpoints or "").split(",") if p]
        if not paths: raise ValueError("generalize mode requires --checkpoints")
        result = generalize(args.run, paths,
                            tuple(int(x) for x in args.bins.split(",")),
                            tuple(int(x) for x in args.gate_phases.split(",")))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
