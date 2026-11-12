"""Research prototype: repository condensed MPC with a 25-state predictor.

Adds three normalized actuator rates to the existing 22 states. Uses only
declared nominal actuator parameters and observed actuator positions/history.
The original source module is loaded in a PRIVATE namespace and its dimension
constant changed there; its QP, weights, constraints and solve routine are used
unchanged. This does not mutate the repository module or production files.
"""
from pathlib import Path
import importlib.util
import numpy as np
import casadi as ca


class SecondOrderDynamics:
    nx = 25

    def __init__(self, dt=.1, fin_gain_scale=1., fin_tau_scale=1., fins_only=False):
        from simulator.model import MISMATCHED_PLANT as p, MASS, M_EFF, D_COEFF, ELEVATOR_COEFF, RUDDER_COEFF
        from simulator.batch_mpc import CONTROL_SCALE
        if not np.isfinite([dt,fin_gain_scale,fin_tau_scale]).all() or dt <= 0 or fin_gain_scale <= 0:
            raise ValueError("finite positive model parameters required")
        if abs(dt/.02-round(dt/.02)) > 1e-9 or p.fin_tau_s*fin_tau_scale < .03:
            raise ValueError("dt must use whole 20ms steps and effective fin tau must be at least .03s")
        self.fin_gain_scale, self.fin_tau_scale = fin_gain_scale, fin_tau_scale
        self.fins_only = fins_only
        z, command = ca.SX.sym("z", 25), ca.SX.sym("command", 3)
        theta, residual = ca.SX.sym("theta", 2), ca.SX.sym("residual", 5)
        x, actual, queue, rate = z[:10], z[10:13], z[13:22], z[22:25]
        im = ca.DM(np.linalg.inv(p.mass_scale*M_EFF))
        tau = ca.vertcat(p.thrust_tau_s, p.fin_tau_s*theta[1], p.fin_tau_s*theta[1])
        om = 1/tau

        def body(s, a):
            th, ps, u, v, w, q, r = [s[i] for i in (3, 4, 5, 6, 7, 8, 9)]
            ct, st, cp, sp = ca.cos(th), ca.sin(th), ca.cos(ps), ca.sin(ps)
            vel = ca.vertcat(cp*ct*u-sp*v+cp*st*w, sp*ct*u+cp*v+sp*st*w, -st*u+ct*w)
            force = ca.vertcat(p.thrust_gain*a[0], 0, 0, 0, 0)
            force += u*u*theta[0]*(p.elevator_gain*ca.DM(ELEVATOR_COEFF)*a[1]
                                     + p.rudder_gain*ca.DM(RUDDER_COEFF)*a[2])
            c = p.mass_scale*MASS*ca.vertcat(-r*v+q*w, r*u, -q*u, 0, 0)
            acc = im@(force-c-p.damping_scale*u*ca.DM(D_COEFF)@s[5:10])+residual
            return ca.vertcat(vel, q, r/ct, acc)

        for _ in range(round(dt/.02)):
            target = queue[6:9]
            queue = ca.vertcat(command, queue[:6])
            rate = rate+.02*(om**2*(target-actual)-2*om*rate)
            new_actual = actual+.02*rate
            if fins_only:
                new_actual[0] = actual[0]+(1-ca.exp(-.02/tau[0]))*(target[0]-actual[0])
                rate[0] = 0
            actual = new_actual
            physical = ca.DM(CONTROL_SCALE)*actual
            k1 = body(x, physical); k2 = body(x+.01*k1, physical)
            k3 = body(x+.01*k2, physical); k4 = body(x+.02*k3, physical)
            x = x+.02*(k1+2*k2+2*k3+k4)/6
        end = ca.vertcat(x, actual, queue, rate)
        self.function = ca.Function("gs12_second_order25", [z, command, theta, residual],
            [end, ca.jacobian(end, z), ca.jacobian(end, command)])

    def evaluate(self, states, controls, model_parameters=None, residual_acceleration=None):
        states, controls = np.asarray(states), np.asarray(controls)
        if states.shape[-1] != 25 or controls.shape[:-1] != states.shape[:-1]:
            raise ValueError("25-state and matching command arrays required")
        shape = states.shape[:-1]
        theta = np.broadcast_to(np.asarray(model_parameters if model_parameters is not None else
            [self.fin_gain_scale, self.fin_tau_scale]), (*shape, 2)).reshape(-1, 2)
        residual = np.broadcast_to(np.asarray(residual_acceleration if residual_acceleration is not None
            else np.zeros(5)), (*shape, 5)).reshape(-1, 5)
        values = [self.function(s, u, th, d) for s, u, th, d in zip(
            states.reshape(-1, 25), controls.reshape(-1, 3), theta, residual)]
        return tuple(np.stack([np.asarray(v[i]).reshape(tail) for v in values]).reshape(*shape, *tail)
                     for i, tail in enumerate(((25,), (25, 25), (25, 3))))


class CausalRateObserver:
    """Reconstruct current rate from two actuator-position observations.

    For each nominal second-order channel: [a+,r+] = H[a,r]+b over
    five substeps. H01 != 0 allows the *previous* rate to be inferred from
    the newly observed a+, then H10/H11 propagate it to current rate.
    The force term b uses the previously issued command and observed queue.
    No true plant tau/delay/rate or future plant state is read.
    """
    def __init__(self, fin_tau_scale=1., fins_only=False):
        from simulator.model import MISMATCHED_PLANT as p
        self.tau = np.array([p.thrust_tau_s, p.fin_tau_s*fin_tau_scale, p.fin_tau_s*fin_tau_scale])
        if not np.isfinite(self.tau).all() or np.any(self.tau < .03):
            raise ValueError("observer effective tau must be finite and at least .03s")
        self.fins_only = fins_only
        self.H = []
        for tau in self.tau:
            om, h = 1/tau, .02
            a = np.array([[1-h*h*om*om, h*(1-2*h*om)], [-h*om*om, 1-2*h*om]])
            self.H.append(np.linalg.matrix_power(a, 5))
        active = self.H[1:] if fins_only else self.H
        if any(abs(h[0, 1]) < 1e-6 for h in active):
            raise ValueError("two-sample actuator-rate reconstruction is singular or poorly conditioned")

    def estimate(self, previous_actual, current_actual, previous_queue, issued_command):
        a, rate = np.asarray(previous_actual).copy(), np.zeros(3)
        queue = np.asarray(previous_queue).reshape(3, 3).copy()
        om = 1/self.tau
        for _ in range(5):
            target = queue[2].copy()
            queue = np.vstack((issued_command, queue[:2]))
            rate += .02*(om**2*(target-a)-2*om*rate)
            a += .02*rate
        estimate = np.zeros(3)
        for j, h in enumerate(self.H):
            previous_rate = (current_actual[j]-a[j])/h[0, 1]
            estimate[j] = rate[j]+h[1, 1]*previous_rate
        if self.fins_only:
            estimate[0] = 0.
        return estimate


def make_second_order_controller(model, repo, horizon=35, fin_gain_scale=1.,
                                 fin_tau_scale=1., fins_only=False):
    path = Path(repo)/"aav_mpc/condensed_mpc.py"
    spec = importlib.util.spec_from_file_location("gs12_private_condensed25", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.AUGMENTED_NX = 25
    controller = module.CondensedActuatorMPC(model, 1, horizon=horizon, dt=.1)
    controller.dynamics = SecondOrderDynamics(.1, fin_gain_scale, fin_tau_scale, fins_only)
    return controller
