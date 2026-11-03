"""Independent plant replay checks for the actuator-aware prediction model."""
import numpy as np

from simulator.batch_mpc import BatchedUUVModel
from simulator.model import DT, MISMATCHED_PLANT
from simulator.rti_symbolic import ActuatorAwareCpuDynamics


def test_augmented_five_substeps_match_torch_plant_with_pending_queue():
    model = BatchedUUVModel("cpu")
    dynamics = ActuatorAwareCpuDynamics(.1)
    hull = np.array([10., -4., 280., .04, .1, 6., .02, -.03, .01, .02])
    actuator = np.array([.1, .02, -.03])
    queue = np.array([[.3, .1, -.2], [.2, -.1, .4], [.1, .2, -.3]])
    command = np.array([.7, -.25, .45])
    predicted = dynamics.evaluate(np.r_[hull, actuator, queue.ravel()][None, None],
                                  command[None, None])[0][0, 0]

    import torch
    state = model.tensor(hull)[None]
    actual = model.tensor(actuator)
    history = model.tensor(queue)
    parameters = model.tensor([MISMATCHED_PLANT.mass_scale,
        MISMATCHED_PLANT.damping_scale, MISMATCHED_PLANT.thrust_gain,
        MISMATCHED_PLANT.elevator_gain, MISMATCHED_PLANT.rudder_gain])[None]
    alpha = model.tensor([-np.expm1(-DT/MISMATCHED_PLANT.thrust_tau_s),
                          -np.expm1(-DT/MISMATCHED_PLANT.fin_tau_s),
                          -np.expm1(-DT/MISMATCHED_PLANT.fin_tau_s)])
    for _ in range(5):
        target = history[2].clone()
        history = torch.cat((model.tensor(command)[None], history[:-1]), 0)
        actual = actual + alpha * (target-actual)
        state = model.rk4(state, (actual*model.control_scale)[None], parameters)
    expected = np.r_[state.numpy().ravel(), actual.numpy(), history.numpy().ravel()]
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-10)


def test_fin_gain_oracle_prediction_matches_scaled_torch_plant():
    import torch

    model = BatchedUUVModel("cpu")
    hull = np.array([10., -4., 280., .04, .1, 15., .02, -.03, .01, .02])
    actual = np.array([.1, .02, -.03])
    queue = np.array([[.3, .1, -.2], [.2, -.1, .4], [.1, .2, -.3]])
    command = np.array([.7, -.25, .45])
    parameters = model.tensor([MISMATCHED_PLANT.mass_scale,
        MISMATCHED_PLANT.damping_scale, MISMATCHED_PLANT.thrust_gain,
        .9*MISMATCHED_PLANT.elevator_gain, .9*MISMATCHED_PLANT.rudder_gain])[None]
    for tau_scale in (1., 1.69):
        dynamics = ActuatorAwareCpuDynamics(.1, fin_gain_scale=.9,
                                            fin_tau_scale=tau_scale)
        predicted = dynamics.evaluate(np.r_[hull, actual, queue.ravel()][None, None],
                                      command[None, None])[0][0, 0]
        state = model.tensor(hull)[None]
        actuator = model.tensor(actual)
        history = model.tensor(queue)
        alpha = model.tensor([-np.expm1(-DT/MISMATCHED_PLANT.thrust_tau_s),
                              -np.expm1(-DT/(MISMATCHED_PLANT.fin_tau_s*tau_scale)),
                              -np.expm1(-DT/(MISMATCHED_PLANT.fin_tau_s*tau_scale))])
        for _ in range(5):
            target = history[2].clone()
            history = torch.cat((model.tensor(command)[None], history[:-1]), 0)
            actuator = actuator + alpha*(target-actuator)
            state = model.rk4(state, (actuator*model.control_scale)[None], parameters)
        expected = np.r_[state.numpy().ravel(), actuator.numpy(), history.numpy().ravel()]
        np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-10)


def test_fixed_target_replay_matches_target_actuator_and_midpoint_integrator():
    from types import SimpleNamespace

    import torch

    from aav_mpc.h15_target_domain import SecondOrderActuator, midpoint_two_substeps
    from aav_mpc.offline_feasibility import SecondOrderActuatorReplay, target_midpoint_step

    plant = MISMATCHED_PLANT
    replay_actuator = SecondOrderActuatorReplay(plant)
    model = BatchedUUVModel("cpu")
    tau = model.tensor([[plant.thrust_tau_s, plant.fin_tau_s, plant.fin_tau_s]])
    holder = SimpleNamespace(actuator=model.tensor([[0., 0., 0.]]), tau=tau)
    target_actuator = SecondOrderActuator(holder)
    queue = model.tensor(np.zeros((3, 3)))
    state_np = np.array([10., -4., 280., .04, .1, 15., .02, -.03, .01, .02])
    state_torch = model.tensor(state_np)[None]
    parameters = model.tensor([[plant.mass_scale, plant.damping_scale,
                                plant.thrust_gain, plant.elevator_gain,
                                plant.rudder_gain]])
    commands = np.array([[8e5, .1, -.2], [1.2e6, -.3, .4],
                         [9e5, .2, .1], [1.4e6, -.4, -.2]])
    for k in range(16):
        command = commands[k % len(commands)]
        actual_np = replay_actuator.step(command)
        target = queue[2].clone()[None]
        queue = torch.cat((model.tensor(command)[None], queue[:-1]), 0)
        holder.actuator = target_actuator(target, DT, torch.ones(1, dtype=torch.bool))
        np.testing.assert_allclose(actual_np, holder.actuator.numpy().ravel(),
                                   rtol=1e-12, atol=1e-10)
        state_np = target_midpoint_step(state_np, actual_np, plant)
        state_torch = midpoint_two_substeps(model, state_torch,
                                            holder.actuator, parameters)
        np.testing.assert_allclose(state_np, state_torch.numpy().ravel(),
                                   rtol=1e-12, atol=1e-10)


def test_fixed_target_symbolic_stage_matches_replay():
    from aav_mpc.offline_feasibility import (SecondOrderActuatorReplay,
                                             stage_function, target_midpoint_step)

    plant = MISMATCHED_PLANT
    hull = np.array([10., -4., 280., .04, .1, 15., .02, -.03, .01, .02])
    normalized = np.array([.7, .2, -.3])
    physical = normalized * np.array([1_500_000., np.pi / 4, np.pi / 4])
    function = stage_function(.1, plant, target_domain=True)
    predicted = np.asarray(function(np.r_[hull, np.zeros(15)], normalized)[0]).ravel()
    actuator = SecondOrderActuatorReplay(plant)
    for _ in range(5):
        actual = actuator.step(physical)
        hull = target_midpoint_step(hull, actual, plant)
    expected = np.r_[hull, actuator.actual,
                     np.tile(physical, 3), actuator.rate]
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-10)


def test_offline_sampled_plant_matches_online_reset_draw():
    import json
    from pathlib import Path

    import torch

    from aav_mpc.controller import EndpointExtendedCondensedEnv
    from aav_mpc.h15_target_domain import install_target_domain
    from aav_mpc.offline_feasibility import (randomized_plant_for_seed,
                                             realized_target_schedule)
    from simulator.batch_path_env import PathBank

    seed = 101041
    plant = randomized_plant_for_seed(seed)
    bank = PathBank([Path("data/datasets/h15_mpc_replication/replication_coupled_a.csv")],
                    "cpu")
    bank.speed[:] = 15.
    bank.first_state[:, 5] = 15.
    bank.durations[:] = bank.lengths / 15.
    env = EndpointExtendedCondensedEnv(bank, batch_size=1, device="cpu",
        horizon=35, backend="rti_cpu", seed=seed, randomize_plant=True,
        current_max_mps=1., current_profile="harmonic", fsp_enabled=False,
        attitude_pitch_weight=100., attitude_yaw_weight=30.,
        preview_mode="multiscale", yaw_reward="course")
    env.reset(torch.zeros(1, dtype=torch.long))
    expected = np.array([plant.mass_scale, plant.damping_scale,
                         plant.thrust_gain, plant.elevator_gain, plant.rudder_gain])
    np.testing.assert_allclose(env.parameters[0].numpy(), expected, rtol=1e-12)
    np.testing.assert_array_equal(env.delay_steps[0].numpy(),
                                  [int(np.ceil(plant.thrust_delay_s/DT-1e-12)),
                                   int(np.ceil(plant.fin_delay_s/DT-1e-12)),
                                   int(np.ceil(plant.fin_delay_s/DT-1e-12))])
    np.testing.assert_allclose(env.tau[0].numpy(),
                               [plant.thrust_tau_s, plant.fin_tau_s, plant.fin_tau_s],
                               rtol=1e-12)
    task = json.loads(Path("aav_mpc/configs/h15.json").read_text())
    current, disturbance = realized_target_schedule(
        Path("data/datasets/h15_mpc_replication/replication_coupled_a.csv"),
        task, seed, 5)
    install_target_domain(env, fin_gain_scale=.9)
    env.step(torch.zeros(1, 3))
    np.testing.assert_allclose(env.current_ned[0].numpy(), current[-1],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(env.disturbance[0].numpy(), disturbance[-1],
                               rtol=0, atol=1e-12)


def test_randomized_four_step_delay_symbolic_target_matches_replay():
    from aav_mpc.offline_feasibility import (SecondOrderActuatorReplay,
        queue_slots, randomized_plant_for_seed, stage_function,
        target_midpoint_step)

    plant = randomized_plant_for_seed(101042)
    assert queue_slots(plant) == 4
    hull = np.array([10., -4., 280., .04, .1, 15., .02, -.03, .01, .02])
    normalized = np.array([.7, .2, -.3])
    physical = normalized*np.array([1_500_000., np.pi/4, np.pi/4])
    predicted = np.asarray(stage_function(.1, plant, target_domain=True)(
        np.r_[hull, np.zeros(3+3*queue_slots(plant)+3)], normalized)[0]).ravel()
    actuator = SecondOrderActuatorReplay(plant)
    for _ in range(5):
        actual = actuator.step(physical)
        hull = target_midpoint_step(hull, actual, plant)
    expected = np.r_[hull, actuator.actual,
                     np.tile(physical, queue_slots(plant)), actuator.rate]
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-10)


def test_realized_target_symbolic_stage_matches_exogenous_replay():
    import json
    from dataclasses import replace
    from pathlib import Path

    from aav_mpc.offline_feasibility import (SecondOrderActuatorReplay,
        queue_slots, randomized_plant_for_seed, realized_target_schedule,
        stage_function, target_midpoint_step)

    seed = 101041
    base = randomized_plant_for_seed(seed)
    plant = replace(base, elevator_gain=.9*base.elevator_gain,
                    rudder_gain=.9*base.rudder_gain)
    task = json.loads(Path("aav_mpc/configs/h15.json").read_text())
    current, disturbance = realized_target_schedule(
        Path("data/datasets/h15_mpc_replication/replication_coupled_a.csv"),
        task, seed, 5)
    hull = np.array([10., -4., 280., .04, .1, 15., .02, -.03, .01, .02])
    normalized = np.array([.7, .2, -.3])
    physical = normalized*np.array([1_500_000., np.pi/4, np.pi/4])
    predicted = np.asarray(stage_function(.1, plant, target_domain=True,
        exogenous=True)(np.r_[hull, np.zeros(3+3*queue_slots(plant)+3)],
                        normalized, current.T, disturbance.T)[0]).ravel()
    actuator = SecondOrderActuatorReplay(plant)
    for k in range(5):
        actual = actuator.step(physical)
        hull = target_midpoint_step(hull, actual, plant,
                                    current[k], disturbance[k])
    expected = np.r_[hull, actuator.actual,
                     np.tile(physical, queue_slots(plant)), actuator.rate]
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-10)
