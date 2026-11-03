"""Target-domain plant update: two-step midpoint hull and second-order fins.

The controller keeps its original symbolic/linear MPC and FSP prediction.
Only the environment's plant integration and actuator update are replaced.
This is a numerical/actuator sim-to-sim stress, not an independent physics model.
"""
import torch


def midpoint_two_substeps(model, state, control, parameters=None,
                          disturbance=None, current=None, dt=.02):
    half = dt / 2.
    x = state
    for _ in range(2):
        k1 = model.derivative(x, control, parameters, disturbance, current)
        k2 = model.derivative(x + half*k1/2., control,
                              parameters, disturbance, current)
        x = x + half*k2
    return x


class SecondOrderActuator:
    def __init__(self, env, tau_scale=1.):
        self.env = env
        self.tau_scale = float(tau_scale)
        self.rate = torch.zeros_like(env.actuator)

    def __call__(self, target, dt, running):
        tau = self.env.tau.clone()
        tau[:, 1:] *= self.tau_scale
        omega = 1. / tau.clamp_min(.03)
        acceleration = omega.square()*(target-self.env.actuator) - 2.*omega*self.rate
        next_rate = self.rate + dt*acceleration
        self.rate = torch.where(running[:, None], next_rate, self.rate)
        return self.env.actuator + dt*self.rate


def install_target_domain(env, fin_gain_scale=.9, integrator=True, second_order=True,
                          fin_tau_scale=1., fin_delay_scale=1.):
    if not 0 < fin_gain_scale <= 1:
        raise ValueError("fin gain scale must lie in (0, 1]")
    if not 0 < fin_tau_scale <= 2:
        raise ValueError("fin time-constant scale must lie in (0, 2]")
    if not 0 < fin_delay_scale <= 1.5:
        raise ValueError("fin delay scale must lie in (0, 1.5]")
    env.parameters[:, 3:5] *= fin_gain_scale
    if fin_delay_scale != 1.:
        env.delay_steps[:, 1:] = torch.ceil(
            env.delay_steps[:, 1:]*fin_delay_scale).long()
    if integrator:
        env.plant_integrator = lambda state, control, parameters=None, disturbance=None, current=None, dt=.02: (
            midpoint_two_substeps(env.model, state, control, parameters,
                                  disturbance, current, dt))
    if second_order:
        env.actuator_update = SecondOrderActuator(env, fin_tau_scale)
