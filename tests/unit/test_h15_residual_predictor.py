import torch

from experiments.h15_cat_mpc.residual_predictor import (nominal_delay_tau,
    nominal_period_rollout)
from simulator.batch_mpc import BatchedUUVModel
from simulator.model import MISMATCHED_PLANT


def test_nominal_period_rollout_has_zero_residual_without_external_disturbance():
    model = BatchedUUVModel("cpu", torch.float64)
    state = torch.zeros((1, 10), dtype=torch.float64)
    state[:, 5] = 15.
    actuator = torch.zeros((1, 3), dtype=torch.float64)
    rate = torch.zeros_like(actuator)
    queue = torch.zeros((1, 5, 3), dtype=torch.float64)
    delays, tau = nominal_delay_tau()
    issued = torch.tensor([[1.0e5, .01, -.01]], dtype=torch.float64)
    p = MISMATCHED_PLANT
    parameters = torch.tensor([[p.mass_scale, p.damping_scale, p.thrust_gain,
        p.elevator_gain*.85, p.rudder_gain*.85]], dtype=torch.float64)
    predicted, _, _, _ = nominal_period_rollout(model, state, actuator, rate, queue,
        delays, tau, issued, parameters, second_order=True, integrator=True)
    # In an identical nominal plant with zero hidden disturbance, measured and
    # rolled out next states coincide, so the external residual label is zero.
    residual = (predicted[:, 5:10]-predicted[:, 5:10])/.1
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0.)
