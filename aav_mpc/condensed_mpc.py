"""CPU condensed RTI-QP for the 22-state actuator-aware UUV model.

State increments are eliminated using the linearized dynamics. The QP has
only 3*N command variables rather than (22+3)*N multiple-shooting variables.
This is the same one-iteration local RTI approximation and tracking objective
as simulator.rti_mpc, with exact box constraints on issued commands.
"""
import copy
import numpy as np
import scipy.sparse as sparse
import osqp
import torch

from simulator.batch_mpc import BatchedLinearMPC
from simulator.rti_symbolic import AUGMENTED_NX, ActuatorAwareCpuDynamics


class CondensedActuatorMPC:
    def __init__(self, model, batch_size, horizon=25, dt=.1,
                 attitude_pitch_weight=100., attitude_yaw_weight=30.,
                 fallback_mode="hold"):
        if fallback_mode not in ("hold", "center_fins"):
            raise ValueError("fallback mode must be hold or center_fins")
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
        self.fallback_mode = fallback_mode
        self.max_normalized_slew = None
        self.screen_scenarios = None
        self.screen_path = None
        self.screen_progress = 0.
        self.sqp_iterations = 1

    def set_sqp_iterations(self, count):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
            raise ValueError("SQP iteration count must be an integer in [1, 5]")
        self.sqp_iterations = count

    def set_command_slew_limits(self, limits):
        if limits is None:
            self.max_normalized_slew = None
            return
        values = np.asarray(limits, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("command slew limits require three positive finite values")
        self.max_normalized_slew = values.copy()

    def set_screen_scenarios(self, scenarios, pitch_limits_deg, speed_limits_mps,
                             path=None):
        if scenarios is None:
            self.screen_scenarios = None
            self.screen_path = None
            return
        pitch_limits = np.asarray(pitch_limits_deg, dtype=float)
        speed_limits = np.asarray(speed_limits_mps, dtype=float)
        for limits in (pitch_limits, speed_limits):
            if limits.shape != (self.horizon,) or not np.isfinite(limits).all() or np.any(limits <= 0):
                raise ValueError("screen limits require one positive finite value per stage")
        pairs = np.asarray(scenarios, dtype=float)
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs) or
                not np.isfinite(pairs).all() or np.any(pairs <= 0)):
            raise ValueError("screen scenarios require positive gain and lag pairs")
        self.screen_scenarios = ([ActuatorAwareCpuDynamics(self.dt, gain, tau)
                                  for gain, tau in pairs], np.deg2rad(pitch_limits),
                                 speed_limits)
        self.screen_path = path

    def set_screen_progress(self, progress):
        if not np.isfinite(progress) or progress < 0:
            raise ValueError("screen progress must be finite and nonnegative")
        self.screen_progress = float(progress)

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
        constraints = sparse.eye(dim, format="csc")
        if self.max_normalized_slew is not None:
            slew = np.tile(self.max_normalized_slew, n)
            constraints = sparse.vstack((constraints, sparse.csc_matrix(difference)),
                                        format="csc")
            lower = np.r_[lower, -slew-rate_nominal]
            upper = np.r_[upper, slew-rate_nominal]
        if self.screen_scenarios is not None:
            dynamics, pitch_limits, speed_limits = self.screen_scenarios
            state_rows, state_lower, state_upper = [], [], []
            for scenario in dynamics:
                predicted = state.copy()
                sensitivity = np.zeros((nx, dim))
                local_path = copy.copy(self.screen_path) if self.screen_path is not None else None
                if local_path is not None:
                    local_path.progress = self.screen_progress
                for k in range(n):
                    f, a, b = scenario.evaluate(predicted[None, None],
                                                 plan[k][None, None])
                    predicted = f[0, 0]
                    sensitivity = a[0, 0] @ sensitivity
                    sensitivity[:, 3*k:3*k+3] += b[0, 0]
                    state_rows.append(sensitivity[3].copy())
                    state_lower.append(-pitch_limits[k]-predicted[3])
                    state_upper.append(pitch_limits[k]-predicted[3])
                    if local_path is None:
                        pitch_ref, yaw_ref = reference[k, 3:5]
                        tangent = np.array([np.cos(pitch_ref)*np.cos(yaw_ref),
                                            np.cos(pitch_ref)*np.sin(yaw_ref),
                                            -np.sin(pitch_ref)])
                    else:
                        local_path.advance(predicted[:3])
                        tangent = local_path.tangent_at(local_path.progress)
                    output, offset = BatchedLinearMPC._tracking_output(
                        self, self.model.tensor(predicted[:10])[None], True)
                    matrix = np.pad(output[0].numpy(), ((0, 0), (0, nx-10)))
                    speed = float(tangent @ (matrix[5:8] @ predicted +
                                             offset[0, 5:8].numpy()))
                    state_rows.append(tangent @ matrix[5:8] @ sensitivity)
                    state_lower.append(-speed_limits[k]-(speed-15.))
                    state_upper.append(speed_limits[k]-(speed-15.))
            constraints = sparse.vstack(
                (constraints, sparse.csc_matrix(np.asarray(state_rows))), format="csc")
            lower = np.r_[lower, state_lower]
            upper = np.r_[upper, state_upper]
        solver.setup(P=sparse.csc_matrix((hessian+hessian.T)/2), q=gradient,
                     A=constraints, l=lower, u=upper,
                     verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                     max_iter=10000, polishing=False)
        solution = solver.solve()
        valid = solution.info.status_val in (1, 2) and np.isfinite(solution.x).all()
        if valid:
            plan = np.clip(plan + solution.x.reshape(n, nu),
                           [0., -1., -1.], [1., 1., 1.])
            if self.max_normalized_slew is not None:
                prior = previous.copy()
                for k in range(n):
                    low = np.maximum([0., -1., -1.], prior-self.max_normalized_slew)
                    high = np.minimum([1., 1., 1.], prior+self.max_normalized_slew)
                    plan[k] = np.clip(plan[k], low, high)
                    prior = plan[k]
        return plan, trajectory, valid, solution.info

    def solve(self, state, reference):
        if state.shape != (self.batch_size, AUGMENTED_NX) or reference.shape != (self.batch_size, self.horizon, 8):
            raise ValueError("invalid state/reference dimensions")
        shifted = torch.cat((self.plan[:, 1:], self.plan[:, -1:]), 1).numpy()
        previous = self.previous.numpy()
        outputs = []
        for i in range(self.batch_size):
            plan = shifted[i]
            for _ in range(self.sqp_iterations):
                plan, trajectory, valid, info = self._one(
                    state[i].numpy(), reference[i].numpy(), plan, previous[i])
                if not valid:
                    break
            self.plan[i] = self.model.tensor(plan)
            self.states[i] = self.model.tensor(trajectory)
            self.last_qp_valid[i] = bool(valid)
            self.last_primal_residual[i] = info.prim_res
            self.last_dual_residual[i] = info.dual_res
            self.last_status[i] = info.status
            if valid:
                self.previous[i] = self.plan[i, 0]
            elif self.fallback_mode == "center_fins":
                self.previous[i] = torch.stack((
                    self.previous[i, 0],
                    self.plan.new_zeros(()),
                    self.plan.new_zeros(()),
                ))
            outputs.append(self.previous[i] * self.model.control_scale)
        return torch.stack(outputs), self.last_qp_valid.clone()
