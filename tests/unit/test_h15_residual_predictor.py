import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aav_mpc.h15_target_domain import SecondOrderActuator
from experiments.h15_cat_mpc.residual_predictor import (nominal_delay_tau,
    nominal_period_rollout)
from experiments.h15_cat_mpc.run_episode import GateEnv
from experiments.h15_cat_mpc.train_predictor import residual_for
from simulator.batch_mpc import BatchedUUVModel
from simulator.model import MISMATCHED_PLANT


def _independent_target_step(model, state, actuator, parameters):
    """Reference implementation of target midpoint integration for this test."""
    # Keep this local so the check does not compare nominal_period_rollout with
    # the same helper it calls internally.
    x = state
    disturbance = torch.zeros((1, 5), dtype=state.dtype)
    for _ in range(2):
        k1 = model.derivative(x, actuator, parameters, disturbance, None)
        k2 = model.derivative(x + .005*k1, actuator, parameters, disturbance, None)
        x = x + .01*k2
    return x


def _independent_period(model, state, actuator, rate, queue, issued, theta):
    p = MISMATCHED_PLANT
    parameters = torch.tensor([[p.mass_scale, p.damping_scale, p.thrust_gain,
        p.elevator_gain*theta[0], p.rudder_gain*theta[0]]], dtype=torch.float64)
    plant = SimpleNamespace(actuator=actuator.clone(), tau=torch.tensor(
        [[p.thrust_tau_s, p.fin_tau_s, p.fin_tau_s]], dtype=torch.float64))
    update = SecondOrderActuator(plant, theta[1])
    update.rate[:] = rate
    state, queue = state.clone(), queue.clone()
    # The plant delay is 50 ms rounded up to three 20 ms updates.
    for _ in range(5):
        target = queue[:, 2].clone()
        queue = torch.cat((issued[:, None], queue[:, :-1]), 1)
        plant.actuator = update(target, .02, torch.tensor([True]))
        state = _independent_target_step(model, state, plant.actuator, parameters)
    return state, plant.actuator, update.rate, queue


def _trace_row(state, after, actuator, queue, issued, collected_theta, collected_rate):
    delay, tau = nominal_delay_tau(fin_tau_scale=collected_theta[1])
    arrays = {"state_before": state[0], "state_after": after[0],
              "actuator_before": actuator[0], "queue_before": queue[0],
              "issued_command": issued[0], "theta_label_before": collected_theta,
              "theta_used": collected_theta, "nominal_delay_steps": delay[0],
              "nominal_tau": tau[0], "nominal_rate_before": collected_rate[0]}
    return {**{f"{key}_json": json.dumps(np.asarray(value).tolist())
               for key, value in arrays.items()},
            "second_order_actuator": "True", "integrator": "True"}


def test_nominal_rollout_matches_independent_target_actuator_for_each_tau_scale():
    """The residual baseline must use theta's fin tau scale exactly once.

    The reference below advances the target-domain actuator and queue directly,
    instead of comparing the rollout implementation with itself. This catches
    the historical effective-tau-times-scale error.
    """
    model = BatchedUUVModel("cpu", torch.float64)
    state = torch.zeros((1, 10), dtype=torch.float64)
    state[:, 5] = 15.
    issued = torch.tensor([[1.0e5, .01, -.01]], dtype=torch.float64)
    p = MISMATCHED_PLANT
    parameters = torch.tensor([[p.mass_scale, p.damping_scale, p.thrust_gain,
        p.elevator_gain*.85, p.rudder_gain*.85]], dtype=torch.float64)
    for scale in (.5, 1., 2., 3.):
        state_before = state.clone()
        actuator = torch.tensor([[2.e4, .02, -.03]], dtype=torch.float64)
        rate = torch.tensor([[.1, -.2, .15]], dtype=torch.float64)
        queue = torch.zeros((1, 5, 3), dtype=torch.float64)
        delays, tau = nominal_delay_tau(fin_tau_scale=scale)
        predicted = nominal_period_rollout(model, state_before, actuator, rate, queue,
            delays, tau, issued, parameters, second_order=True, integrator=True)

        class Plant:
            pass
        plant = Plant()
        plant.tau = torch.tensor([[p.thrust_tau_s, p.fin_tau_s, p.fin_tau_s]],
                                  dtype=torch.float64)
        plant.actuator = actuator.clone()
        actuator_update = SecondOrderActuator(plant, scale)
        actuator_update.rate[:] = rate
        actual_state = state_before.clone()
        actual_queue = queue.clone()
        for _ in range(5):
            delayed = actual_queue.gather(1, (delays-1).clamp_min(0)[:, None]).squeeze(1)
            target = torch.where(delays > 0, delayed, issued)
            actual_queue = torch.cat((issued[:, None], actual_queue[:, :-1]), 1)
            plant.actuator = actuator_update(target, .02, torch.tensor([True]))
            actual_state = _independent_target_step(model, actual_state, plant.actuator,
                                                    parameters)
        torch.testing.assert_close(predicted[0], actual_state, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(predicted[1], plant.actuator, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(predicted[2], actuator_update.rate, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(predicted[3], actual_queue, atol=1e-11, rtol=1e-11)


def test_relabel_uses_replay_theta_and_rate_instead_of_collection_values():
    model = BatchedUUVModel("cpu", torch.float64)
    state = torch.zeros((1, 10), dtype=torch.float64)
    state[:, 5] = 15.
    actuator = torch.tensor([[2.e4, .02, -.03]], dtype=torch.float64)
    queue = torch.zeros((1, 5, 3), dtype=torch.float64)
    issued = torch.tensor([[1.e5, .03, -.02]], dtype=torch.float64)
    replay_rate = torch.tensor([[.1, -.2, .15]], dtype=torch.float64)
    collected_rate = torch.zeros((1, 3), dtype=torch.float64)
    collected_theta, replay_theta = (.85, 2.), (.95, .7)
    actual = _independent_period(model, state, actuator, replay_rate, queue,
                                 issued, replay_theta)
    row = _trace_row(state, actual[0], actuator, queue, issued,
                     collected_theta, collected_rate)
    residual, next_rate = residual_for(model, row, replay_theta, replay_rate)
    np.testing.assert_allclose(residual, np.zeros(5), atol=1e-11, rtol=0.)
    torch.testing.assert_close(next_rate, actual[2], atol=1e-11, rtol=1e-11)
    wrong_residual, _ = residual_for(model, row, collected_theta, collected_rate)
    assert np.max(np.abs(wrong_residual)) > 1e-5
    # Recorded rate is valid provenance, but never replaces recursive replay rate.
    row["nominal_rate_before_json"] = json.dumps([100., 10., -10.])
    same_residual, same_rate = residual_for(model, row, replay_theta, replay_rate)
    np.testing.assert_array_equal(same_residual, residual)
    torch.testing.assert_close(same_rate, next_rate)
    # A v3 trace that already applied scale twice is rejected before relabeling.
    row["nominal_tau_json"] = json.dumps([.1, .32, .32])
    with pytest.raises(ValueError, match="collection theta actuator contract"):
        residual_for(model, row, replay_theta, replay_rate)


def test_online_offline_labels_match_independent_plant_over_step_and_reversal():
    model = BatchedUUVModel("cpu", torch.float64)
    state = torch.zeros((1, 10), dtype=torch.float64)
    state[:, 5] = 15.
    env = SimpleNamespace(model=model, device="cpu", dtype=torch.float64,
        state=state, actuator=torch.zeros((1, 3), dtype=torch.float64),
        command_history=torch.zeros((1, 5, 3), dtype=torch.float64),
        nominal_actuator_rate=torch.zeros((1, 3), dtype=torch.float64),
        theta_hat=np.array([.85, 2.]), alpha=np.zeros(2), d_forecast=np.zeros((35, 5)),
        pending=None, finished=torch.tensor([False]), completed_rows=[],
        residual_history=[], period_index=0, progress=torch.zeros(1),
        mpc=SimpleNamespace(plan=torch.zeros((1, 35, 3), dtype=torch.float64)),
        actuator_update=True, plant_integrator=True,
        _errors=lambda: (torch.zeros(1),)*6)
    plant_rate = torch.zeros((1, 3), dtype=torch.float64)
    replay_rate = torch.zeros_like(plant_rate)
    for sign in (1., 1., -1., -1.):
        issued = torch.tensor([[1.e5, sign*.03, -sign*.02]], dtype=torch.float64)
        env.base_command = issued
        GateEnv._start_pending(env)
        env.pending["issued_command"] = issued[0].numpy().copy()
        env.pending["low_level_compensation"] = np.zeros(3)
        actual = _independent_period(model, env.state, env.actuator, plant_rate,
                                     env.command_history, issued, env.theta_hat)
        env.state, env.actuator, plant_rate, env.command_history = actual
        GateEnv._complete_pending(env)
        row = env.completed_rows[-1]
        replay_residual, replay_rate = residual_for(model, row, env.theta_hat, replay_rate)
        np.testing.assert_allclose(row["residual_body_acc"], np.zeros(5), atol=1e-11, rtol=0.)
        np.testing.assert_allclose(replay_residual, row["residual_body_acc"], atol=1e-11, rtol=0.)
        torch.testing.assert_close(replay_rate, plant_rate, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(env.nominal_actuator_rate, plant_rate, atol=1e-11, rtol=1e-11)
