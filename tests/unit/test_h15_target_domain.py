import torch

from aav_mpc.h15_target_domain import install_target_domain


class DummyEnv:
    def __init__(self):
        self.parameters = torch.ones(1, 5, dtype=torch.float64)
        self.delay_steps = torch.tensor([[3, 3, 3]], dtype=torch.long)
        self.tau = torch.tensor([[.1, .08, .08]], dtype=torch.float64)
        self.actuator = torch.zeros(1, 3, dtype=torch.float64)


def test_target_domain_scales_fin_gain_delay_and_time_constant_only():
    env = DummyEnv()
    install_target_domain(env, fin_gain_scale=.9, integrator=False,
                          second_order=True, fin_tau_scale=2.,
                          fin_delay_scale=1.5)
    # Registration must not mutate pre-reset values; the reset hook applies
    # the target after randomization and before the first solve.
    torch.testing.assert_close(env.parameters[:, 3:], torch.ones(1, 2, dtype=torch.float64))
    env._after_reset_parameters(torch.tensor([True]))
    torch.testing.assert_close(env.parameters[:, :3], torch.ones(1, 3, dtype=torch.float64))
    torch.testing.assert_close(env.parameters[:, 3:], torch.full((1, 2), .9,
                                                                 dtype=torch.float64))
    assert env.delay_steps.tolist() == [[3, 5, 5]]
    target = torch.tensor([[0., .1, .1]], dtype=torch.float64)
    result = env.actuator_update(target, .02, torch.tensor([True]))
    expected = .02 * .02 * (1. / (.08*2.)) ** 2 * .1
    torch.testing.assert_close(result, torch.tensor([[0., expected, expected]],
                                                    dtype=torch.float64))


def test_target_domain_reapplies_to_each_reset_snapshot():
    env = DummyEnv()
    install_target_domain(env, fin_gain_scale=.9, integrator=False,
                          second_order=True)
    mask = torch.tensor([True])
    env._after_reset_parameters(mask)
    torch.testing.assert_close(env.parameters[:, 3:], torch.full((1, 2), .9,
                                                                 dtype=torch.float64))
    first = env._target_domain_reset_snapshot["parameters_fin_before"]
    env.parameters[:, 3:] = 1.
    env._after_reset_parameters(mask)
    torch.testing.assert_close(env.parameters[:, 3:], torch.full((1, 2), .9,
                                                                 dtype=torch.float64))
    assert env._target_domain_reset_snapshot["parameters_fin_before"] == first
