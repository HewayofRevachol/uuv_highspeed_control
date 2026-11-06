"""25-state hybrid predictor and four-to-two channel MPC bridge.

State: hull 10, measured normalized actuators 3, causal thrust-rate estimate
and two nominal fin-rate states 3, newest-first issued command queue 9.
The predictor always uses a second-order thrust and first-order fins.
"""
from __future__ import annotations

import time

import casadi as ca
import numpy as np
import scipy.sparse as sparse
import osqp
import torch

from aav_mpc.condensed_mpc import CondensedActuatorMPC
from simulator.batch_mpc import BatchedLinearMPC, CONTROL_SCALE
from simulator.model import (D_COEFF, DT, ELEVATOR_COEFF, MASS, M_EFF,
                             MISMATCHED_PLANT, RUDDER_COEFF)

NX = 25
THETA_SAFE = np.array([.85, 2.], dtype=float)
NU_INDEX = np.array([5, 6, 7, 8, 9])


def bounded_gate(previous, target, slew=.25):
    previous, target = np.asarray(previous, float), np.asarray(target, float)
    if previous.shape != (4,) or target.shape != (4,) or not np.isfinite(target).all():
        return np.zeros(4), True
    return np.clip(previous + np.clip(np.clip(target, 0., 1.)-previous,
                                      -slew, slew), 0., 1.), False


def mix_gate(alpha, theta_hat, residual):
    """Mix outside the old two-branch interface, then pass alpha=[1,1]."""
    alpha = np.asarray(alpha, float)
    theta_hat = np.asarray(theta_hat, float)
    residual = np.asarray(residual, float)
    if alpha.shape != (4,) or theta_hat.shape != (2,) or residual.shape != (5,):
        raise ValueError("expected alpha[4], theta[2], residual[5]")
    if not all(np.isfinite(v).all() for v in (alpha, theta_hat, residual)):
        raise ValueError("nonfinite gate input")
    if np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("gate outside [0,1]")
    theta = THETA_SAFE + alpha[0]*(theta_hat-THETA_SAFE)
    d = residual * alpha[[1, 3, 2, 2, 3]]
    return theta, d


def hybrid_function(dt=.1):
    count = round(dt/DT)
    if count < 1 or abs(count*DT-dt) > 1e-12:
        raise ValueError("dt must be a multiple of .02 s")
    p = MISMATCHED_PLANT
    z, command = ca.SX.sym("z", NX), ca.SX.sym("cmd", 3)
    theta, residual = ca.SX.sym("theta", 2), ca.SX.sym("d", 5)
    x, actual, rate, queue = z[:10], z[10:13], z[13:16], z[16:]
    inverse_mass = ca.DM(np.linalg.inv(p.mass_scale*M_EFF))
    delay = [int(np.ceil(p.thrust_delay_s/DT-1e-12)),
             int(np.ceil(p.fin_delay_s/DT-1e-12)),
             int(np.ceil(p.fin_delay_s/DT-1e-12))]

    def derivative(state, actuator):
        pitch, yaw = state[3], state[4]
        u, v, w, q, r = (state[i] for i in range(5, 10))
        ct, st, cp, sp = ca.cos(pitch), ca.sin(pitch), ca.cos(yaw), ca.sin(yaw)
        pos = ca.vertcat(cp*ct*u-sp*v+cp*st*w,
                            sp*ct*u+cp*v+sp*st*w, -st*u+ct*w)
        physical = ca.diag(ca.DM(CONTROL_SCALE)) @ actuator
        force = ca.vertcat(p.thrust_gain*physical[0], 0, 0, 0, 0)
        force += u*u*theta[0]*(p.elevator_gain*ca.DM(ELEVATOR_COEFF)*physical[1]
                                   + p.rudder_gain*ca.DM(RUDDER_COEFF)*physical[2])
        coriolis = p.mass_scale*MASS*ca.vertcat(-r*v+q*w, r*u, -q*u, 0, 0)
        acc = inverse_mass @ (force-coriolis-p.damping_scale*u*ca.DM(D_COEFF)@state[5:10])
        return ca.vertcat(pos, q, r/ct, acc+residual)

    for _ in range(count):
        target = ca.vertcat(*[command[j] if delay[j] == 0 else
                              queue[3*(delay[j]-1)+j] for j in range(3)])
        queue = ca.vertcat(command, queue[:-3])
        # Critically damped thrust; rate is normalized actuator units per second.
        omega = 1/max(p.thrust_tau_s, .03)
        rate0 = rate[0] + DT*(omega*omega*(target[0]-actual[0])-2*omega*rate[0])
        thrust = actual[0] + DT*rate0
        fins = [actual[j] + (-ca.expm1(-DT/(p.fin_tau_s*theta[1])))*
                (target[j]-actual[j]) for j in (1, 2)]
        actual = ca.vertcat(thrust, *fins)
        rate = ca.vertcat(rate0, 0, 0)
        for _ in range(2):
            half = DT/2
            k1 = derivative(x, actual)
            k2 = derivative(x+half*k1/2, actual)
            x = x+half*k2
    end = ca.vertcat(x, actual, rate, queue)
    return ca.Function("gs12_hybrid", [z, command, theta, residual],
                       [end, ca.jacobian(end, z), ca.jacobian(end, command)])


class HybridDynamics:
    nx = NX

    def __init__(self, dt=.1):
        self.function = hybrid_function(dt)
        self.fin_gain_scale, self.fin_tau_scale = THETA_SAFE

    def evaluate(self, states, commands, model_parameters=None,
                 residual_acceleration=None):
        states, commands = np.asarray(states), np.asarray(commands)
        if states.shape[:-1] != commands.shape[:-1] or states.shape[-1] != NX or commands.shape[-1] != 3:
            raise ValueError("hybrid state/command shape mismatch")
        theta = np.broadcast_to(THETA_SAFE if model_parameters is None else model_parameters,
                                (*states.shape[:-1], 2)).reshape(-1, 2)
        d = np.broadcast_to(np.zeros(5) if residual_acceleration is None else residual_acceleration,
                            (*states.shape[:-1], 5)).reshape(-1, 5)
        flat_z, flat_u = states.reshape(-1, NX), commands.reshape(-1, 3)
        result = [self.function(x, u, th, di) for x, u, th, di in zip(flat_z, flat_u, theta, d)]
        shape = states.shape[:-1]
        return (np.stack([np.asarray(v[0]).reshape(NX) for v in result]).reshape(*shape, NX),
                np.stack([np.asarray(v[1]).reshape(NX, NX) for v in result]).reshape(*shape, NX, NX),
                np.stack([np.asarray(v[2]).reshape(NX, 3) for v in result]).reshape(*shape, NX, 3))


class ThrustRateObserver:
    """Causal measured-thrust derivative; no plant hidden rate is read."""
    def __init__(self, smoothing=.5):
        self.smoothing = smoothing
        self.previous = None
        self.rate = 0.

    def reset(self):
        self.previous, self.rate = None, 0.

    def update(self, normalized_thrust, dt=.1):
        value = float(normalized_thrust)
        if self.previous is not None:
            self.rate = self.smoothing*(value-self.previous)/dt + (1-self.smoothing)*self.rate
        self.previous = value
        return self.rate


class HybridMPC(CondensedActuatorMPC):
    """Condensed RTI with the same QP weights, now on the 25-state model."""
    def __init__(self, model, batch_size=1, horizon=35, dt=.1):
        super().__init__(model, batch_size, horizon=horizon, dt=dt,
                         attitude_pitch_weight=100., attitude_yaw_weight=30.)
        self.dynamics = HybridDynamics(dt)
        self.states = torch.zeros(batch_size, horizon+1, NX, dtype=model.dtype, device=model.device)

    def _one(self, state, reference, plan, previous, theta, residual):
        n, dim = self.horizon, 3*self.horizon
        trajectory = np.zeros((n+1, NX)); trajectory[0] = state
        influence = np.zeros((NX, dim))
        weight_u = np.tile([.15, .08, .08], n)
        difference = np.eye(dim)
        for k in range(1, n):
            difference[3*k:3*k+3, 3*(k-1):3*k] = -np.eye(3)
        rate_weight = np.tile([1.8, .35, .35], n)
        rate_nominal = difference@plan.ravel(); rate_nominal[:3] -= previous
        hessian = np.diag(1e-6+weight_u) + difference.T@(rate_weight[:, None]*difference)
        gradient = weight_u*plan.ravel() + difference.T@(rate_weight*rate_nominal)
        state_weight = np.array([3., 3., 3., 100., 30., 5., 5., 5.])
        for k in range(n):
            f, a, b = self.dynamics.evaluate(trajectory[k][None, None],
                plan[k][None, None], model_parameters=theta,
                residual_acceleration=residual[k])
            trajectory[k+1] = f[0, 0]
            influence = a[0, 0]@influence
            influence[:, 3*k:3*k+3] += b[0, 0]
            output, offset = BatchedLinearMPC._tracking_output(
                self, self.model.tensor(trajectory[k+1, :10])[None], True)
            matrix = np.pad(output[0].numpy(), ((0, 0), (0, NX-10)))
            error = matrix@trajectory[k+1] + offset[0].numpy()-reference[k]
            error[4] = np.arctan2(np.sin(error[4]), np.cos(error[4]))
            sensitivity = matrix@influence
            w = state_weight*(2.5 if k == n-1 else 1.)
            hessian += sensitivity.T@(w[:, None]*sensitivity)
            gradient += sensitivity.T@(w*error)
        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix((hessian+hessian.T)/2), q=gradient,
                     A=sparse.eye(dim, format="csc"),
                     l=np.tile([0., -1., -1.], n)-plan.ravel(),
                     u=np.ones(dim)-plan.ravel(), verbose=False,
                     eps_abs=1e-5, eps_rel=1e-5, max_iter=10000)
        solution = solver.solve()
        valid = solution.info.status_val in (1, 2) and solution.x is not None and np.isfinite(solution.x).all()
        if valid:
            plan = np.clip(plan+solution.x.reshape(n, 3), [0., -1., -1.], [1., 1., 1.])
        return plan, trajectory, valid, solution.info

    def solve(self, state, reference, alpha=None, theta_hat=None, d_hat=None):
        if state.shape != (self.batch_size, NX) or reference.shape != (self.batch_size, self.horizon, 8):
            raise ValueError("HybridMPC requires state [B,25] and reference [B,N,8]")
        alpha = np.asarray([1., 1.] if alpha is None else alpha, float)
        if alpha.shape != (2,) or not np.allclose(alpha, 1.):
            raise ValueError("four-channel mixing must occur in bridge before solve")
        theta = np.asarray(THETA_SAFE if theta_hat is None else theta_hat, float)
        d = np.asarray(np.zeros((self.horizon, 5)) if d_hat is None else d_hat, float)
        d = np.broadcast_to(d, (self.batch_size, self.horizon, 5))
        if theta.shape != (2,) or not np.isfinite(theta).all() or not np.isfinite(d).all():
            raise ValueError("nonfinite model input")
        shifted = torch.cat((self.plan[:, 1:], self.plan[:, -1:]), 1).numpy()
        outputs = []
        for i in range(self.batch_size):
            plan, trajectory, valid, info = self._one(state[i].numpy(), reference[i].numpy(),
                shifted[i], self.previous[i].numpy(), theta, d[i])
            self.plan[i] = self.model.tensor(plan)
            self.states[i] = self.model.tensor(trajectory)
            self.last_qp_valid[i] = bool(valid)
            self.last_primal_residual[i] = info.prim_res
            self.last_dual_residual[i] = info.dual_res
            self.last_status[i] = info.status
            if valid:
                self.previous[i] = self.plan[i, 0]
            outputs.append(self.previous[i]*self.model.control_scale)
        return torch.stack(outputs), self.last_qp_valid.clone()


def finite_difference_check(dynamics, state, command, theta=THETA_SAFE, eps=1e-6):
    f, a, b = dynamics.evaluate(state[None, None], command[None, None], theta)
    a_fd, b_fd = np.empty((NX, NX)), np.empty((NX, 3))
    for j in range(NX):
        shift = np.zeros(NX); shift[j] = eps
        plus = dynamics.evaluate((state+shift)[None, None], command[None, None], theta)[0][0, 0]
        minus = dynamics.evaluate((state-shift)[None, None], command[None, None], theta)[0][0, 0]
        a_fd[:, j] = (plus-minus)/(2*eps)
    for j in range(3):
        shift = np.zeros(3); shift[j] = eps
        plus = dynamics.evaluate(state[None, None], (command+shift)[None, None], theta)[0][0, 0]
        minus = dynamics.evaluate(state[None, None], (command-shift)[None, None], theta)[0][0, 0]
        b_fd[:, j] = (plus-minus)/(2*eps)
    return {"propagation_finite": bool(np.isfinite(f).all()),
            "A_max_abs": float(np.max(np.abs(a[0, 0]-a_fd))),
            "B_max_abs": float(np.max(np.abs(b[0, 0]-b_fd)))}
