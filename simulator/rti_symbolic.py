"""CasADi nominal dynamics for the existing 10-state UUV.

The command is normalized exactly as in MPCController.  The plant used by the
environment may still have current, parameter mismatch and actuator lag.
"""

import numpy as np

from simulator.batch_mpc import CONTROL_SCALE
from simulator.model import (D_COEFF, ELEVATOR_COEFF, MASS, M_EFF,
                             RUDDER_COEFF, DT, MISMATCHED_PLANT)


NX = 10
NU = 3
AUGMENTED_NX = 22  # hull 10, actual controls 3, newest-first pending commands 3x3


def discrete_function(dt=.1, substeps=1):
    """Return CasADi Function (x, normalized_u) -> (f_d, df/dx, df/du)."""
    import casadi as ca

    if dt <= 0 or substeps < 1:
        raise ValueError("dt and substeps must be positive")
    x = ca.SX.sym("x", NX)
    command = ca.SX.sym("command", NU)
    inverse_mass = ca.DM(np.linalg.inv(M_EFF))
    damping_matrix = ca.DM(D_COEFF)
    elevator = ca.DM(ELEVATOR_COEFF)
    rudder = ca.DM(RUDDER_COEFF)

    def derivative(state):
        theta, psi = state[3], state[4]
        u, v, w, q, r = (state[i] for i in range(5, 10))
        ct, st, cp, sp = ca.cos(theta), ca.sin(theta), ca.cos(psi), ca.sin(psi)
        position_dot = ca.vertcat(cp*ct*u-sp*v+cp*st*w,
                                  sp*ct*u+cp*v+sp*st*w, -st*u+ct*w)
        coriolis = MASS*ca.vertcat(-r*v+q*w, r*u, -q*u, 0, 0)
        damping = u*ca.mtimes(damping_matrix, state[5:10])
        physical = ca.vertcat(*(float(CONTROL_SCALE[i])*command[i]
                                for i in range(NU)))
        force = ca.vertcat(physical[0], 0, 0, 0, 0)
        force += u*u*(elevator*physical[1]+rudder*physical[2])
        nu_dot = ca.mtimes(inverse_mass, force-coriolis-damping)
        return ca.vertcat(position_dot, q, r/ct, nu_dot)

    step_dt = dt/substeps
    next_state = x
    for _ in range(substeps):
        k1 = derivative(next_state)
        k2 = derivative(next_state+step_dt*k1/2)
        k3 = derivative(next_state+step_dt*k2/2)
        k4 = derivative(next_state+step_dt*k3)
        next_state = next_state+step_dt*(k1+2*k2+2*k3+k4)/6
    return ca.Function("uuv_fd_jac", [x, command],
                       [next_state, ca.jacobian(next_state, x),
                        ca.jacobian(next_state, command)])


class CpuCasadiDynamics:
    def __init__(self, dt=.1, substeps=1):
        self.function = discrete_function(dt, substeps)

    def evaluate(self, states, controls):
        """Accept NumPy [B,N,nx] and [B,N,nu]; return f, A, B."""
        states, controls = np.asarray(states), np.asarray(controls)
        if states.shape[:-1] != controls.shape[:-1] or states.shape[-1] != NX or controls.shape[-1] != NU:
            raise ValueError("states/controls require matching batch and horizon")
        count = int(np.prod(states.shape[:-1]))
        flat_x, flat_u = states.reshape(count, NX), controls.reshape(count, NU)
        values = [self.function(flat_x[i], flat_u[i]) for i in range(count)]
        f = np.stack([np.asarray(item[0]).reshape(NX) for item in values])
        a = np.stack([np.asarray(item[1]).reshape(NX, NX) for item in values])
        b = np.stack([np.asarray(item[2]).reshape(NX, NU) for item in values])
        shape = states.shape[:-1]
        return f.reshape(*shape, NX), a.reshape(*shape, NX, NX), b.reshape(*shape, NX, NU)


def actuator_aware_function(dt=.1):
    """Fixed-parameter plant with measured actuator and queued command state.

    Normalized commands and actuator states keep the QP better scaled. The
    three pending slots retain exactly the delay-relevant part of the plant's
    five-slot history; older commands cannot affect future actuator targets.
    The controller uses declared mismatch parameters, never sampled plant
    parameters or delay_steps from the simulator.
    """
    import casadi as ca

    count = round(dt / DT)
    if count < 1 or abs(count * DT - dt) > 1e-12:
        raise ValueError("dt must be an integer multiple of plant DT")
    p = MISMATCHED_PLANT
    z = ca.SX.sym("z", AUGMENTED_NX)
    command = ca.SX.sym("command", NU)
    x, actual, queue = z[:NX], z[NX:NX+NU], z[NX+NU:]
    inverse_mass = ca.DM(np.linalg.inv(p.mass_scale * M_EFF))
    damping = ca.DM(D_COEFF)
    elevator = ca.DM(ELEVATOR_COEFF)
    rudder = ca.DM(RUDDER_COEFF)
    scale = ca.DM(CONTROL_SCALE)
    delay = np.array([int(np.ceil(p.thrust_delay_s/DT-1e-12)),
                      int(np.ceil(p.fin_delay_s/DT-1e-12))] * 1)
    delays = [delay[0], delay[1], delay[1]]
    alpha = [-np.expm1(-DT/t) if t else 1. for t in
             (p.thrust_tau_s, p.fin_tau_s, p.fin_tau_s)]

    def body_derivative(state, physical):
        theta, psi = state[3], state[4]
        u, v, w, q, r = (state[i] for i in range(5, 10))
        ct, st, cp, sp = ca.cos(theta), ca.sin(theta), ca.cos(psi), ca.sin(psi)
        position_dot = ca.vertcat(cp*ct*u-sp*v+cp*st*w,
                                  sp*ct*u+cp*v+sp*st*w, -st*u+ct*w)
        coriolis = p.mass_scale*MASS*ca.vertcat(-r*v+q*w, r*u, -q*u, 0, 0)
        force = ca.vertcat(p.thrust_gain*physical[0], 0, 0, 0, 0)
        force += u*u*(p.elevator_gain*elevator*physical[1] +
                      p.rudder_gain*rudder*physical[2])
        acceleration = inverse_mass @ (force-coriolis-p.damping_scale*u*damping@state[5:10])
        return ca.vertcat(position_dot, q, r/ct, acceleration)

    for _ in range(count):
        target = ca.vertcat(*[
            command[j] if delays[j] == 0 else queue[3*(delays[j]-1)+j]
            for j in range(NU)])
        queue = ca.vertcat(command, queue[:-NU])
        actual = actual+ca.vertcat(*[alpha[j]*(target[j]-actual[j]) for j in range(NU)])
        physical = ca.vertcat(*[scale[j]*actual[j] for j in range(NU)])
        k1 = body_derivative(x, physical)
        k2 = body_derivative(x+DT*k1/2, physical)
        k3 = body_derivative(x+DT*k2/2, physical)
        k4 = body_derivative(x+DT*k3, physical)
        x = x+DT*(k1+2*k2+2*k3+k4)/6
    end = ca.vertcat(x, actual, queue)
    return ca.Function("uuv_aug_fd_jac", [z, command],
                       [end, ca.jacobian(end, z), ca.jacobian(end, command)])


class ActuatorAwareCpuDynamics:
    nx = AUGMENTED_NX

    def __init__(self, dt=.1):
        self.function = actuator_aware_function(dt)

    def evaluate(self, states, controls):
        states, controls = np.asarray(states), np.asarray(controls)
        if states.shape[:-1] != controls.shape[:-1] or states.shape[-1] != self.nx or controls.shape[-1] != NU:
            raise ValueError("invalid augmented state/control shape")
        count = int(np.prod(states.shape[:-1]))
        values = [self.function(states.reshape(count, self.nx)[i],
                                controls.reshape(count, NU)[i]) for i in range(count)]
        f = np.stack([np.asarray(item[0]).reshape(self.nx) for item in values])
        a = np.stack([np.asarray(item[1]).reshape(self.nx, self.nx) for item in values])
        b = np.stack([np.asarray(item[2]).reshape(self.nx, NU) for item in values])
        shape = states.shape[:-1]
        return (f.reshape(*shape, self.nx), a.reshape(*shape, self.nx, self.nx),
                b.reshape(*shape, self.nx, NU))
