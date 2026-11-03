import torch

from high_speed_experiments.recovery_mpcc import ProgressEnv, screen, speed_components
from simulator.batch_mpc import BatchedUUVModel
from simulator.batch_path_env import PathBank
from simulator.rti_mpc import BatchedRTIMPC


def test_disconnected_feasible_speeds_are_preserved():
    assert speed_components([6, 9, 12, 15, 18],
                            [True, True, False, True, True]) == [[6, 9], [15, 18]]


def test_recovery_screen_uses_estimates_and_restores_controller():
    bank = PathBank(["DDAT_trajectories/val/constant_turn_01.csv"], "cpu")
    env = ProgressEnv(bank, batch_size=1, device="cpu", horizon=4,
                      backend="osqp", randomize_plant=False,
                      current_max_mps=0., fsp_enabled=True)
    env.reset(torch.zeros(1, dtype=torch.long))
    before_plan = env.mpc.plan.clone()
    before_command = env.base_command.clone()
    first = screen(env, [6., 9.], steps=2)
    assert torch.equal(env.mpc.plan, before_plan)
    assert torch.equal(env.base_command, before_command)
    env.parameters *= 1.3
    env.current_ned += .5
    env.delay_steps += 2
    env.tau += .2
    second = screen(env, [6., 9.], steps=2)
    assert first == second


def test_constrained_rti_qp_adds_path_attitude_and_slew_rows():
    model = BatchedUUVModel("cpu")
    mpc = BatchedRTIMPC(model, 1, horizon=3, backend="rti_cpu",
                        corridor_lateral_m=8., corridor_depth_m=5.,
                        pitch_limit_deg=50., max_normalized_slew=.3)
    state = model.tensor([[0., 0., 0., 0., 0., 18., 0., 0., 0., 0.]])
    reference = torch.zeros(1, 3, 8, dtype=model.dtype)
    reference[0, :, 0] = model.tensor([1.8, 3.6, 5.4])
    reference[0, :, 5] = 18.
    _, _, A, lower, upper = mpc.build_qp(state, reference)
    assert A.shape == (1, 3 * 13 + 3 * 6, 3 * 13)
    assert lower.shape == upper.shape == (1, A.shape[1])
    previous = mpc.previous.clone()
    _, ok = mpc.solve(state, reference)
    assert bool(ok[0])
    assert (mpc.plan[:, 0] - previous).abs().max() <= .3 + 1e-6
    assert (mpc.plan[:, 1:] - mpc.plan[:, :-1]).abs().max() <= .3 + 1e-6


def test_soft_rti_path_constraints_report_unavoidable_slack():
    model = BatchedUUVModel("cpu")
    mpc = BatchedRTIMPC(model, 1, horizon=3, backend="rti_cpu",
                        corridor_lateral_m=.1, corridor_depth_m=.1,
                        pitch_limit_deg=5., max_normalized_slew=.3,
                        soft_path_penalty=100.)
    state = model.tensor([[0., 0., 0., 0., 0., 15., 0., 0., 0., 0.]])
    reference = torch.zeros(1, 3, 8, dtype=model.dtype)
    reference[0, :, 0] = model.tensor([1.5, 3., 4.5])
    reference[0, :, 1] = 20.
    reference[0, :, 5] = 15.
    _, ok = mpc.solve(state, reference)
    assert bool(ok[0])
    assert mpc.last_status == ["solved"]
    assert (mpc.last_slack[0, :, 0] > 19.).all()
    assert (mpc.last_slack >= 0.).all()
    assert (mpc.plan[:, 1:] - mpc.plan[:, :-1]).abs().max() <= .3 + 1e-6
