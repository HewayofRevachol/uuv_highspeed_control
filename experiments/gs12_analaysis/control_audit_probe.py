"""Read-only production-source probes for GS-12; outputs stay outside repository."""
from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

BASE = Path(__file__).resolve().parent
REPO = BASE.parent / "uuv_highspeed_control"
sys.path.insert(0, str(REPO))

import numpy as np
import torch

from simulator.batch_mpc import BatchedUUVModel
from simulator.model import MISMATCHED_PLANT
from aav_mpc.h15_target_domain import SecondOrderActuator, midpoint_two_substeps
from experiments.h15_cat_mpc.residual_predictor import nominal_delay_tau, nominal_period_rollout
from aav_mpc import diagnose
from experiments.h15_cat_mpc.run_episode import run_episode


def tau_probe():
    model = BatchedUUVModel("cpu")
    theta = (.85, 2.)
    p = MISMATCHED_PLANT
    state = model.tensor([[0., 0., 100., 0., 0., 15., 0., 0., 0., 0.]])
    actuator = model.tensor([[0., 0., 0.]])
    rate = torch.zeros_like(actuator)
    queue = model.tensor(np.zeros((1, 5, 3)))
    delays, tau = nominal_delay_tau(fin_tau_scale=theta[1])
    issued = model.tensor([[800000., .1, -.1]])
    parameters = model.tensor([[p.mass_scale, p.damping_scale, p.thrust_gain,
                                p.elevator_gain*theta[0], p.rudder_gain*theta[0]]])
    live = SimpleNamespace(actuator=actuator.clone(), tau=tau.clone())
    update = SecondOrderActuator(live, tau_scale=1.)
    actual_state, actual_queue = state.clone(), queue.clone()
    for _ in range(5):
        delayed = actual_queue.gather(1, (delays-1).clamp_min(0)[:, None]).squeeze(1)
        target = torch.where(delays > 0, delayed, issued)
        actual_queue = torch.cat((issued[:, None], actual_queue[:, :-1]), 1)
        live.actuator = update(target, .02, torch.tensor([True]))
        actual_state = midpoint_two_substeps(model, actual_state, live.actuator,
                                             parameters, dt=.02)
    predicted, predicted_actuator, _, _ = nominal_period_rollout(
        model, state, actuator, rate, queue, delays, tau, issued, parameters,
        second_order=True, fin_tau_scale=theta[1], integrator=True)
    single_scaled, single_actuator, _, _ = nominal_period_rollout(
        model, state, actuator, rate, queue, delays, tau, issued, parameters,
        second_order=True, fin_tau_scale=1., integrator=True)
    return {
        "theta": theta,
        "caller_tau_s": tau[0].tolist(),
        "production_label_effective_fin_tau_s": (tau[0, 1:]*theta[1]).tolist(),
        "actual_matched_plant_fin_tau_s": tau[0, 1:].tolist(),
        "issued_physical": issued[0].tolist(),
        "actual_actuator_after": live.actuator[0].tolist(),
        "production_label_actuator_after": predicted_actuator[0].tolist(),
        "production_label_zero_disturbance_residual": ((actual_state-predicted)[0, 5:]/.1).tolist(),
        "single_scale_reference_residual_max": float(((actual_state-single_scaled)/.1).abs().max()),
        "single_scale_reference_actuator_error_max": float((live.actuator-single_actuator).abs().max()),
    }


def diagnose_install_probe():
    result = {}
    held = {}
    original = diagnose.install_target_domain
    original_root = diagnose.ROOT
    def capture(env, **kwargs):
        held["env"] = env
        result["parameters_before_install"] = env.parameters[0].tolist()
        result["delays_before_install"] = env.delay_steps[0].tolist()
        value = original(env, **kwargs)
        result["requested_config"] = value
        result["parameters_after_install"] = env.parameters[0].tolist()
        result["delays_after_install"] = env.delay_steps[0].tolist()
        result["snapshot_present_after_install"] = hasattr(env, "_target_domain_reset_snapshot")
        return value
    diagnose.install_target_domain = capture
    # Only relocate the output-label root; use an absolute input path below.
    diagnose.ROOT = BASE.parent
    try:
        diagnose.run("H15", "condensed_endpoint35", BASE / "diagnose_install_one_period.csv",
                     max_steps=1, path_override=REPO/"data/datasets/h15_mainline_quickstart/train_late_entry.csv",
                     seed_override=71001, randomize_plant=False, current_max=0., process_noise=False,
                     horizon_override=3, target_domain=True, target_components=("fin",),
                     target_fin_gain_scale=.5, target_fin_delay_scale=.5)
        env = held["env"]
        result["parameters_after_one_period"] = env.parameters[0].tolist()
        result["delays_after_one_period"] = env.delay_steps[0].tolist()
        env.reset(torch.zeros(1, dtype=torch.long))
        result["parameters_after_subsequent_reset"] = env.parameters[0].tolist()
        result["delays_after_subsequent_reset"] = env.delay_steps[0].tolist()
    finally:
        diagnose.install_target_domain = original
        diagnose.ROOT = original_root
    return result


def existing_e0_excitation_probe():
    out = BASE / "matched_nominal_e0_existing"
    if (out / "episode.json").exists():
        value = json.loads((out / "episode.json").read_text())
    else:
        value = run_episode(REPO/"data/datasets/h15_mainline_quickstart/train_late_entry.csv",
            71001, out, method="A0", environment_profile="matched_nominal",
            compensation_mode="off", action_profile="step", predictor_mode="zero", max_steps=40)
    commands = np.asarray([r["command"] for r in value["trace"]])
    actuators = np.asarray([r["actuator"] for r in value["strict_substeps"]])
    residuals = np.asarray([r["residual_body_acc"] for r in value["completed_rows"]])
    return {"periods": value["periods"],
            "max_abs_fin_command_rad": float(np.abs(commands[:, 1:]).max()),
            "max_abs_fin_actuator_rad": float(np.abs(actuators[:, 1:]).max()),
            "max_abs_residual": float(np.abs(residuals).max()),
            "step_action_channels": [0]}


if __name__ == "__main__":
    torch.set_num_threads(1)
    report = {"tau_double_scale": tau_probe(), "diagnose_install_order": diagnose_install_probe(),
              "existing_e0_excitation": existing_e0_excitation_probe()}
    report["source_sha256"] = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [REPO/"experiments/h15_cat_mpc/residual_predictor.py",
                     REPO/"experiments/h15_cat_mpc/run_episode.py",
                     REPO/"aav_mpc/diagnose.py", REPO/"aav_mpc/h15_target_domain.py"]}
    (BASE / "control_audit_probe.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
