"""CPU condensed RTI-QP for the 22-state actuator-aware UUV model.

State increments are eliminated using the linearized dynamics. The QP has
only 3*N command variables rather than (22+3)*N multiple-shooting variables.
This is the same one-iteration local RTI approximation and tracking objective
as simulator.rti_mpc, with exact box constraints on issued commands.
"""
import numpy as np
import scipy.sparse as sparse
import osqp
import torch

from simulator.batch_mpc import BatchedLinearMPC
from simulator.rti_symbolic import AUGMENTED_NX, ActuatorAwareCpuDynamics


class CondensedActuatorMPC:
    def __init__(self, model, batch_size, horizon=25, dt=.1,
                 attitude_pitch_weight=100., attitude_yaw_weight=30.):
        self.model, self.batch_size, self.horizon, self.dt = model, batch_size, horizon, dt
        self.backend = "rti_cpu"
        self.tolerance = 1e-5
        self.attitude_pitch_weight = attitude_pitch_weight
        self.attitude_yaw_weight = attitude_yaw_weight
        self.dynamics = ActuatorAwareCpuDynamics(dt)
        self.plan = model.tensor([.6, 0., 0.]).expand(batch_size, horizon, 3).clone()
        self.states = torch.zeros(batch_size, horizon+1, AUGMENTED_NX, dtype=model.dtype,
                                   device=model.device)
        self.previous = self.plan[:, 0].clone()
        self.last_primal_residual = model.tensor(np.zeros(batch_size))
        self.last_dual_residual = model.tensor(np.zeros(batch_size))
        self.last_qp_valid = torch.ones(batch_size, dtype=torch.bool, device=model.device)
        self.last_status = ["not_solved"] * batch_size

    def reset(self, state, mask=None):
        if mask is None:
            mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.model.device)
        self.plan[mask] = self.model.tensor([.6, 0., 0.])
        self.previous[mask] = 0
        self.states[mask] = 0
        self.last_qp_valid[mask] = True
        self.last_primal_residual[mask] = 0
        self.last_dual_residual[mask] = 0

    def _one(self, state, reference, plan, previous):
        n, nx, nu = self.horizon, AUGMENTED_NX, 3
        dim = n * nu
        trajectory = np.zeros((n+1, nx))
        trajectory[0] = state
        influence = np.zeros((nx, dim))
        hessian = np.eye(dim) * 1e-6
        gradient = np.zeros(dim)
        state_weight = np.array([3., 3., 3., self.attitude_pitch_weight,
                                 self.attitude_yaw_weight, 5., 5., 5.])
        input_weight = np.tile([.15, .08, .08], n)
        hessian += np.diag(input_weight)
        gradient += input_weight * plan.ravel()
        rate_weight = np.tile([1.8, .35, .35], n)
        difference = np.eye(dim)
        for k in range(1, n):
            difference[3*k:3*k+3, 3*(k-1):3*k] = -np.eye(3)
        rate_nominal = difference @ plan.ravel()
        rate_nominal[:3] -= previous
        hessian += difference.T @ (rate_weight[:, None] * difference)
        gradient += difference.T @ (rate_weight * rate_nominal)
        for k in range(n):
            f, a, b = self.dynamics.evaluate(trajectory[k][None, None], plan[k][None, None])
            trajectory[k+1] = f[0, 0]
            influence = a[0, 0] @ influence
            influence[:, 3*k:3*k+3] += b[0, 0]
            x_torch = self.model.tensor(trajectory[k+1, :10])[None]
            output, offset = BatchedLinearMPC._tracking_output(self, x_torch, True)
            matrix = np.pad(output[0].numpy(), ((0, 0), (0, nx-10)))
            error = matrix @ trajectory[k+1] + offset[0].numpy() - reference[k]
            error[4] = np.arctan2(np.sin(error[4]), np.cos(error[4]))
            sensitivity = matrix @ influence
            weight = state_weight * (2.5 if k == n-1 else 1.)
            hessian += sensitivity.T @ (weight[:, None] * sensitivity)
            gradient += sensitivity.T @ (weight * error)
        solver = osqp.OSQP()
        lower = np.tile([0., -1., -1.], n) - plan.ravel()
        upper = np.ones(dim) - plan.ravel()
        solver.setup(P=sparse.csc_matrix((hessian+hessian.T)/2), q=gradient,
                     A=sparse.eye(dim, format="csc"), l=lower, u=upper,
                     verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                     max_iter=10000, polishing=False)
        solution = solver.solve()
        valid = solution.info.status_val in (1, 2) and np.isfinite(solution.x).all()
        if valid:
            plan = np.clip(plan + solution.x.reshape(n, nu),
                           [0., -1., -1.], [1., 1., 1.])
        return plan, trajectory, valid, solution.info

    def solve(self, state, reference):
        if state.shape != (self.batch_size, AUGMENTED_NX) or reference.shape != (self.batch_size, self.horizon, 8):
            raise ValueError("invalid state/reference dimensions")
        shifted = torch.cat((self.plan[:, 1:], self.plan[:, -1:]), 1).numpy()
        previous = self.previous.numpy()
        outputs = []
        for i in range(self.batch_size):
            plan, trajectory, valid, info = self._one(
                state[i].numpy(), reference[i].numpy(), shifted[i], previous[i])
            self.plan[i] = self.model.tensor(plan)
            self.states[i] = self.model.tensor(trajectory)
            self.last_qp_valid[i] = bool(valid)
            self.last_primal_residual[i] = info.prim_res
            self.last_dual_residual[i] = info.dual_res
            self.last_status[i] = info.status
            if valid:
                self.previous[i] = self.plan[i, 0]
            outputs.append(self.previous[i] * self.model.control_scale)
        return torch.stack(outputs), self.last_qp_valid.clone()
