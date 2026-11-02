"""CasADi nominal dynamics for the existing 10-state UUV.

The command is normalized exactly as in MPCController.  The plant used by the
environment may still have current, parameter mismatch and actuator lag.
"""

import numpy as np

from simulator.batch_mpc import CONTROL_SCALE
from simulator.model import (D_COEFF, ELEVATOR_COEFF, MASS, M_EFF,
                             RUDDER_COEFF)


NX = 10
NU = 3


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
