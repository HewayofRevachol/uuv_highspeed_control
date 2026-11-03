"""Batched version of the repository's 10-state, three-actuator UUV model/MPC.

All runtime arrays remain on the requested torch device.  This is a dense
QP prototype; it is deliberately separate from the optional sparse
CuSADi/cuDSS implementation described in the design note.
"""

import numpy as np
import torch

from simulator.model import (D_COEFF, DT, ELEVATOR_COEFF, MASS, M_EFF,
                             MAX_RUDDER, RUDDER_COEFF)


CONTROL_SCALE = (1_500_000., MAX_RUDDER, MAX_RUDDER)


class BatchedUUVModel:
    def __init__(self, device, dtype=torch.float64):
        self.device, self.dtype = torch.device(device), dtype
        self.mass = MASS
        self.inverse_mass = self.tensor(np.linalg.inv(M_EFF))
        self.damping = self.tensor(D_COEFF)
        self.elevator = self.tensor(ELEVATOR_COEFF)
        self.rudder = self.tensor(RUDDER_COEFF)
        self.control_scale = self.tensor(CONTROL_SCALE)
        self.lower = self.tensor([0., -1., -1.])
        self.upper = self.tensor([1., 1., 1.])

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def derivative(self, state, control, parameters=None, disturbance=None,
                   current=None):
        """Match simulator.model.derivative for [B,10] states and [B,3] controls.

        parameters columns: mass, damping, thrust, elevator, rudder gains.
        Disturbance is a body acceleration, not an inertial current velocity.
        """
        if parameters is None:
            parameters = torch.ones((*state.shape[:-1], 5), dtype=self.dtype,
                                    device=self.device)
        theta, psi = state[..., 3], state[..., 4]
        u, v, w, q, r = state[..., 5:].unbind(-1)
        ct, st, cp, sp = torch.cos(theta), torch.sin(theta), torch.cos(psi), torch.sin(psi)
        position_dot = torch.stack((cp*ct*u-sp*v+cp*st*w,
                                    sp*ct*u+cp*v+sp*st*w, -st*u+ct*w), -1)
        if current is not None:
            position_dot = position_dot + current
        coriolis = self.mass * parameters[..., :1] * torch.stack(
            (-r*v+q*w, r*u, -q*u, torch.zeros_like(u), torch.zeros_like(u)), -1)
        damping = parameters[..., 1:2] * u[..., None] * (state[..., 5:] @ self.damping.T)
        force = torch.stack((parameters[..., 2]*control[..., 0],
                             torch.zeros_like(u), torch.zeros_like(u),
                             torch.zeros_like(u), torch.zeros_like(u)), -1)
        force = force + u.square()[..., None] * (
            parameters[..., 3:4] * control[..., 1:2] * self.elevator +
            parameters[..., 4:5] * control[..., 2:3] * self.rudder)
        nu_dot = ((force-coriolis-damping) @ self.inverse_mass.T) / parameters[..., :1]
        if disturbance is not None:
            nu_dot = nu_dot + disturbance
        return torch.cat((position_dot, q[..., None], (r/ct)[..., None], nu_dot), -1)

    def rk4(self, state, control, parameters=None, disturbance=None,
            current=None, dt=DT):
        def f(x):
            return self.derivative(x, control, parameters, disturbance, current)
        k1 = f(state)
        k2 = f(state + dt*k1/2)
        k3 = f(state + dt*k2/2)
        k4 = f(state + dt*k3)
        return state + dt*(k1+2*k2+2*k3+k4)/6

    def linearize(self, state, normalized_control, dt):
        """Central differences and second-order discretization as in MPCController."""
        control = normalized_control * self.control_scale
        f0 = self.derivative(state, control)
        a_cols = []
        for i in range(10):
            h = 1e-4 * torch.maximum(torch.ones_like(state[:, i]), state[:, i].abs())
            offset = torch.zeros_like(state)
            offset[:, i] = h
            a_cols.append((self.derivative(state+offset, control) -
                           self.derivative(state-offset, control))/(2*h[:, None]))
        a = torch.stack(a_cols, -1)
        b_cols = []
        for i in range(3):
            offset = torch.zeros_like(control)
            offset[:, i] = 1e-3*self.control_scale[i]
            b_cols.append((self.derivative(state, control+offset) -
                           self.derivative(state, control-offset))/2e-3)
        b = torch.stack(b_cols, -1)
        aa = a @ a
        ad = torch.eye(10, device=self.device, dtype=self.dtype) + dt*a + .5*dt*dt*aa
        bd = dt*b + .5*dt*dt*(a @ b)
        drift = dt*f0 + .5*dt*dt*(a @ f0[..., None]).squeeze(-1)
        c = state + drift - (ad @ state[..., None]).squeeze(-1) - (bd @ normalized_control[..., None]).squeeze(-1)
        return ad, bd, c


class BatchedLinearMPC:
    """Condensed local-linear MPC with batched box ADMM or optional CPU OSQP."""

    def __init__(self, model, batch_size, horizon=10, dt=.1, backend="admm",
                 iterations=80, rho=2., tolerance=1e-3,
                 attitude_pitch_weight=0., attitude_yaw_weight=0.):
        self.model, self.batch_size, self.horizon, self.dt = model, batch_size, horizon, dt
        self.backend, self.iterations, self.rho, self.tolerance = backend, iterations, rho, tolerance
        if (not np.isfinite(attitude_pitch_weight) or attitude_pitch_weight < 0 or
                not np.isfinite(attitude_yaw_weight) or attitude_yaw_weight < 0):
            raise ValueError("attitude tracking weights must be finite and nonnegative")
        self.attitude_pitch_weight = float(attitude_pitch_weight)
        self.attitude_yaw_weight = float(attitude_yaw_weight)
        if batch_size < 1 or horizon < 1 or iterations < 1 or dt <= 0 or rho <= 0:
            raise ValueError("batch_size, horizon, iterations, dt and rho must be positive")
        if backend == "osqp":
            try:
                import osqp  # noqa: F401
                import scipy.sparse  # noqa: F401
            except ImportError as error:
                raise RuntimeError("CPU OSQP backend requires 'osqp' and 'scipy'") from error
            if model.device.type != "cpu":
                raise ValueError("OSQP backend only supports CPU")
            self.last_status = ["not_solved"] * batch_size
        elif backend != "admm":
            raise ValueError("backend must be 'admm' or 'osqp'")
        self.reset(torch.ones(batch_size, 10, device=model.device, dtype=model.dtype))

    def reset(self, state, mask=None):
        count = state.shape[0]
        if not hasattr(self, "previous"):
            self.previous = self.model.tensor([.6, 0., 0.]).expand(count, -1).clone()
            self.plan = self.previous[:, None, :].expand(-1, self.horizon, -1).clone()
            self.last_primal_residual = torch.zeros(count, device=self.model.device,
                                                    dtype=self.model.dtype)
            self.last_dual_residual = torch.zeros_like(self.last_primal_residual)
            self.last_qp_valid = torch.ones(count, device=self.model.device,
                                            dtype=torch.bool)
        elif mask is not None:
            self.previous[mask] = self.model.tensor([.6, 0., 0.])
            self.plan[mask] = self.previous[mask, None, :]
            self.last_primal_residual[mask] = 0
            self.last_dual_residual[mask] = 0
            self.last_qp_valid[mask] = True

    def _tracking_output(self, state, include_attitude=False):
        theta, psi = state[:, 3], state[:, 4]
        u, v, w = state[:, 5], state[:, 6], state[:, 7]
        ct, st, cp, sp = torch.cos(theta), torch.sin(theta), torch.cos(psi), torch.sin(psi)
        b = state.shape[0]
        velocity_start = 5 if include_attitude else 3
        out = torch.zeros(b, velocity_start+3, 10,
                          device=self.model.device, dtype=self.model.dtype)
        out[:, :3, :3] = torch.eye(3, device=self.model.device, dtype=self.model.dtype)
        if include_attitude:
            out[:, 3:5, 3:5] = torch.eye(2, device=self.model.device, dtype=self.model.dtype)
        out[:, velocity_start, 3] = -cp*st*u+cp*ct*w
        out[:, velocity_start+1, 3] = -sp*st*u+sp*ct*w
        out[:, velocity_start+2, 3] = -ct*u-st*w
        out[:, velocity_start, 4] = -sp*ct*u-cp*v-sp*st*w
        out[:, velocity_start+1, 4] = cp*ct*u-sp*v+cp*st*w
        out[:, velocity_start+2, 4] = 0
        out[:, velocity_start, 5:8] = torch.stack((cp*ct, -sp, cp*st), -1)
        out[:, velocity_start+1, 5:8] = torch.stack((sp*ct, cp, sp*st), -1)
        out[:, velocity_start+2, 5:8] = torch.stack((-st, torch.zeros_like(st), ct), -1)
        velocity = self.model.derivative(
            state, torch.zeros(b, 3, device=self.model.device, dtype=self.model.dtype))[:, :3]
        actual = torch.cat((state[:, :5] if include_attitude else state[:, :3],
                            velocity), -1)
        offset = actual - (out @ state[..., None]).squeeze(-1)
        return out, offset

    def build_qp(self, state, reference):
        """Track [position, pitch, yaw, velocity] or legacy [position, velocity]."""
        if reference.shape not in ((self.batch_size, self.horizon, 6),
                                   (self.batch_size, self.horizon, 8)):
            raise ValueError("reference must have shape [batch, horizon, 6 or 8]")
        include_attitude = reference.shape[-1] == 8
        b, n = state.shape[0], self.horizon*3
        ad, bd, c = self.model.linearize(state, self.previous, self.dt)
        output, offset = self._tracking_output(state, include_attitude)
        h = torch.zeros(b, n, n, device=self.model.device, dtype=self.model.dtype)
        g = torch.zeros(b, n, device=self.model.device, dtype=self.model.dtype)
        influence = torch.zeros(b, 10, n, device=self.model.device, dtype=self.model.dtype)
        free = state
        q = self.model.tensor([3, 3, 3, self.attitude_pitch_weight,
                               self.attitude_yaw_weight, 5, 5, 5] if include_attitude
                              else [3, 3, 3, 5, 5, 5])
        input_weight = self.model.tensor([.15, .08, .08])
        rate_weight = self.model.tensor([1.8, .35, .35])
        for k in range(self.horizon):
            influence = ad @ influence
            influence[:, :, 3*k:3*k+3] += bd
            free = (ad @ free[..., None]).squeeze(-1) + c
            oi = output @ influence
            error = (output @ free[..., None]).squeeze(-1) + offset - reference[:, k]
            if include_attitude:
                error[:, 4] = torch.atan2(torch.sin(error[:, 4]), torch.cos(error[:, 4]))
            weight = q * (2.5 if k == self.horizon-1 else 1.)
            h += oi.transpose(-1, -2) @ (weight[None, :, None]*oi)
            g += (oi.transpose(-1, -2) @ (weight*error)[..., None]).squeeze(-1)
            sl = slice(3*k, 3*k+3)
            h[:, sl, sl] += torch.diag(input_weight+rate_weight)
            if k == 0:
                g[:, sl] -= rate_weight*self.previous
            else:
                prev = slice(3*(k-1), 3*k)
                h[:, prev, prev] += torch.diag(rate_weight)
                h[:, sl, prev] -= torch.diag(rate_weight)
                h[:, prev, sl] -= torch.diag(rate_weight)
        return h, g

    def _admm_iterate(self, factors, safe_g, initial_z, low, high):
        z = initial_z.clone()
        y = torch.zeros_like(z)
        for _ in range(self.iterations):
            x = torch.cholesky_solve(
                (-safe_g+self.rho*(z-y))[..., None], factors).squeeze(-1)
            old_z = z
            z = torch.minimum(torch.maximum(x+y, low), high)
            y = y+x-z
        return x, z, old_z

    def _admm_iterate_cuda_graph(self, factors, safe_g, initial_z, low, high):
        # A single environment launches thousands of tiny CUDA operations per
        # solve. Capture the fixed iteration loop once and replay it with new
        # factors, gradient and warm start on subsequent MPC steps.
        if not hasattr(self, "_admm_graph"):
            static_factors = factors.clone()
            static_g = safe_g.clone()
            static_z = initial_z.clone()
            static_low = low.clone()
            static_high = high.clone()
            stream = torch.cuda.Stream(device=self.model.device)
            stream.wait_stream(torch.cuda.current_stream(self.model.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._admm_iterate(
                        static_factors, static_g, static_z, static_low, static_high)
            torch.cuda.current_stream(self.model.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self._admm_iterate(
                    static_factors, static_g, static_z, static_low, static_high)
            self._admm_graph = graph
            self._admm_graph_inputs = (
                static_factors, static_g, static_z, static_low, static_high)
            self._admm_graph_outputs = outputs
        static_factors, static_g, static_z, _, _ = self._admm_graph_inputs
        static_factors.copy_(factors)
        static_g.copy_(safe_g)
        static_z.copy_(initial_z)
        self._admm_graph.replay()
        return self._admm_graph_outputs

    def solve(self, state, reference):
        h, g = self.build_qp(state, reference)
        if self.backend == "osqp":
            import osqp
            import scipy.sparse as sparse
            solutions, primal_residuals, dual_residuals, statuses = [], [], [], []
            for i in range(self.batch_size):
                solver = osqp.OSQP()
                solver.setup(P=sparse.csc_matrix(h[i].numpy()), q=g[i].numpy(),
                             A=sparse.eye(h.shape[-1], format="csc"),
                             l=np.tile([0., -1., -1.], self.horizon),
                             u=np.tile([1., 1., 1.], self.horizon),
                             verbose=False, polishing=False,
                             eps_abs=1e-7, eps_rel=1e-7, max_iter=20_000)
                solver.warm_start(x=self.plan[i].flatten().numpy())
                result = solver.solve()
                statuses.append(result.info.status)
                solutions.append(result.x if result.info.status_val in (1, 2)
                                 else np.full(h.shape[-1], np.nan))
                primal_residuals.append(getattr(result.info, "prim_res", float("inf")))
                dual_residuals.append(getattr(result.info, "dual_res", float("inf")))
            solution = self.model.tensor(np.stack(solutions))
            converged = torch.isfinite(solution).all(-1)
            self.last_qp_valid = torch.isfinite(h).all((-1, -2)) & torch.isfinite(g).all(-1)
            self.last_primal_residual = self.model.tensor(primal_residuals)
            self.last_dual_residual = self.model.tensor(dual_residuals)
            self.last_status = statuses
        else:
            size = h.shape[-1]
            matrix = h + self.rho*torch.eye(size, device=h.device, dtype=h.dtype)
            finite_qp = torch.isfinite(matrix).all((-1, -2)) & torch.isfinite(g).all(-1)
            matrix = torch.where(finite_qp[:, None, None], matrix,
                                 torch.eye(size, device=h.device, dtype=h.dtype))
            safe_g = torch.where(finite_qp[:, None], g, torch.zeros_like(g))
            factors, info = torch.linalg.cholesky_ex(matrix)
            valid = (info == 0) & finite_qp & torch.isfinite(factors).all((-1, -2))
            factors = torch.where(valid[:, None, None], factors,
                                  torch.eye(size, device=h.device, dtype=h.dtype))
            z = torch.cat((self.plan[:, 1:], self.plan[:, -1:]), 1).reshape(self.batch_size, -1).clone()
            low = self.model.lower.repeat(self.horizon)
            high = self.model.upper.repeat(self.horizon)
            if self.model.device.type == "cuda" and self.batch_size == 1:
                x, z, old_z = self._admm_iterate_cuda_graph(
                    factors, safe_g, z, low, high)
            else:
                x, z, old_z = self._admm_iterate(factors, safe_g, z, low, high)
            primal = (x-z).abs().amax(-1)
            dual = (self.rho*(z-old_z)).abs().amax(-1)
            converged = valid & torch.isfinite(z).all(-1) & (primal < self.tolerance) & (dual < self.tolerance)
            self.last_qp_valid = valid
            self.last_primal_residual = torch.where(valid, primal, torch.full_like(primal, float("inf")))
            self.last_dual_residual = torch.where(valid, dual, torch.full_like(dual, float("inf")))
            solution = z
        new_plan = solution.reshape(self.batch_size, self.horizon, 3)
        self.plan = torch.where(converged[:, None, None], new_plan, self.plan)
        self.previous = torch.where(converged[:, None], self.plan[:, 0], self.previous)
        return self.previous*self.model.control_scale, converged
