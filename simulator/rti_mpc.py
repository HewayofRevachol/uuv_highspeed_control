"""One-iteration nonlinear multiple-shooting RTI-SQP controller.

Decision variables are state increments at nodes 1..N and normalized input
increments at stages 0..N-1.  The dynamics are linearized about a shifted
trajectory at every call.  The resulting convex QP has exact input bounds.
"""

import numpy as np
import torch

from simulator.rti_symbolic import NX, NU, AUGMENTED_NX, CpuCasadiDynamics, ActuatorAwareCpuDynamics


class BatchedRTIMPC:
    def __init__(self, model, batch_size, horizon=20, dt=.1, backend="rti_cpu",
                 iterations=400, tolerance=2e-3, cusadi_root=None, build_dir=None,
                 attitude_pitch_weight=0., attitude_yaw_weight=0.,
                 corridor_lateral_m=None, corridor_depth_m=None,
                 pitch_limit_deg=None, max_normalized_slew=None,
                 actuator_aware=False):
        if horizon < 1 or batch_size < 1 or dt <= 0:
            raise ValueError("invalid RTI dimensions or step")
        if backend not in ("rti_cpu", "rti_cuda"):
            raise ValueError("backend must be rti_cpu or rti_cuda")
        if backend == "rti_cpu" and model.device.type != "cpu":
            raise ValueError("rti_cpu requires a CPU model")
        if backend == "rti_cuda" and model.device.type != "cuda":
            raise ValueError("rti_cuda requires a CUDA model")
        if (not np.isfinite(attitude_pitch_weight) or attitude_pitch_weight < 0 or
                not np.isfinite(attitude_yaw_weight) or attitude_yaw_weight < 0):
            raise ValueError("attitude tracking weights must be finite and nonnegative")
        self.attitude_pitch_weight = float(attitude_pitch_weight)
        self.attitude_yaw_weight = float(attitude_yaw_weight)
        constraints = (corridor_lateral_m, corridor_depth_m, pitch_limit_deg,
                       max_normalized_slew)
        if any(v is not None and (not np.isfinite(v) or v <= 0) for v in constraints):
            raise ValueError("RTI state and slew limits must be positive and finite")
        if any(v is not None for v in constraints) and not all(v is not None for v in constraints):
            raise ValueError("all RTI path, attitude and slew limits are required together")
        self.path_constraints = constraints if all(v is not None for v in constraints) else None
        self.model, self.batch_size, self.horizon, self.dt = model, batch_size, horizon, dt
        self.backend, self.iterations, self.tolerance = backend, iterations, tolerance
        if actuator_aware and backend != "rti_cpu":
            raise ValueError("actuator-aware dynamics currently require rti_cpu")
        self.actuator_aware = actuator_aware
        self.nx = AUGMENTED_NX if actuator_aware else NX
        self.dynamics = (ActuatorAwareCpuDynamics(dt) if actuator_aware else
                         CpuCasadiDynamics(dt) if backend == "rti_cpu" else
                         self._cuda_dynamics(cusadi_root, build_dir))
        self.solver = None
        self.reset(torch.zeros(batch_size, self.nx, device=model.device, dtype=model.dtype))

    def _cuda_dynamics(self, cusadi_root, build_dir):
        from simulator.rti_cuda import CudaCasadiDynamics
        return CudaCasadiDynamics(self.batch_size*self.horizon, self.dt,
                                   cusadi_root=cusadi_root, build_dir=build_dir)

    def reset(self, state, mask=None):
        if self.actuator_aware and state.shape[-1] == NX:
            state = torch.cat((state, torch.zeros(state.shape[0], self.nx-NX,
                device=state.device, dtype=state.dtype)), -1)
        if not hasattr(self, "plan"):
            self.plan = self.model.tensor([.6, 0., 0.]).expand(
                self.batch_size, self.horizon, NU).clone()
            self.states = state[:, None, :].expand(-1, self.horizon+1, -1).clone()
            self.previous = self.plan[:, 0].clone()
        elif mask is not None:
            self.plan[mask] = self.model.tensor([.6, 0., 0.])
            self.states[mask] = state[mask, None, :]
            self.previous[mask] = self.plan[mask, 0]

    def _shift_and_rollout(self, state):
        # RTI shift: input sequence is shifted by one stage, then the nominal
        # states are re-integrated to satisfy the nonlinear dynamics initially.
        self.plan = torch.cat((self.plan[:, 1:], self.plan[:, -1:]), 1)
        self.states = torch.cat((self.states[:, 1:], self.states[:, -1:]), 1)
        self.states[:, 0] = state
        for k in range(self.horizon):
            if self.backend == "rti_cpu":
                f, _, _ = self.dynamics.evaluate(
                    self.states[:, k].numpy()[:, None],
                    self.plan[:, k].numpy()[:, None])
                self.states[:, k+1] = self.model.tensor(f[:, 0])
            else:
                f, _, _ = self.dynamics.evaluate(
                    self.states[:, k:k+1], self.plan[:, k:k+1])
                self.states[:, k+1] = f[:, 0]

    def build_qp(self, state, reference):
        """Return P,q,A,l,u for 1/2 dz'P dz+q'dz, l<=A dz<=u."""
        if (state.shape != (self.batch_size, self.nx) or
                reference.shape not in ((self.batch_size, self.horizon, 6),
                                        (self.batch_size, self.horizon, 8))):
            raise ValueError("invalid state or reference shape")
        include_attitude = reference.shape[-1] == 8
        self._shift_and_rollout(state)
        b, n = self.batch_size, self.horizon
        if self.backend == "rti_cpu":
            f, jac_x, jac_u = self.dynamics.evaluate(
                self.states[:, :-1].numpy(), self.plan.numpy())
            f, jac_x, jac_u = (self.model.tensor(x) for x in (f, jac_x, jac_u))
        else:
            f, jac_x, jac_u = self.dynamics.evaluate(
                self.states[:, :-1], self.plan)
        nx = self.nx
        ns, nu, dim = n*nx, n*NU, n*(nx+NU)
        P = torch.zeros(b, dim, dim, device=self.model.device, dtype=self.model.dtype)
        q = torch.zeros(b, dim, device=self.model.device, dtype=self.model.dtype)
        extra_rows = 6*n if self.path_constraints is not None else 0
        A = torch.zeros(b, dim+extra_rows, dim, device=self.model.device, dtype=self.model.dtype)
        lower = torch.zeros(b, dim+extra_rows, device=self.model.device, dtype=self.model.dtype)
        upper = torch.zeros_like(lower)
        state_weight = self.model.tensor([3., 3., 3., self.attitude_pitch_weight,
                                          self.attitude_yaw_weight, 5., 5., 5.] if
                                         include_attitude else [3., 3., 3., 5., 5., 5.])
        input_weight = self.model.tensor([.15, .08, .08])
        rate_weight = self.model.tensor([1.8, .35, .35])
        for k in range(n):
            sx = slice(k*nx, (k+1)*nx)
            su = slice(ns+k*NU, ns+(k+1)*NU)
            row = slice(k*nx, (k+1)*nx)
            A[:, row, sx] = torch.eye(nx, device=self.model.device, dtype=self.model.dtype)
            A[:, row, su] = -jac_u[:, k]
            if k:
                A[:, row, slice((k-1)*nx, k*nx)] = -jac_x[:, k]
            defect = f[:, k]-self.states[:, k+1]
            lower[:, row] = defect
            upper[:, row] = defect
            output, offset = self._tracking_output(self.states[:, k+1], include_attitude)
            tracking_error = (output @ self.states[:, k+1, :, None]).squeeze(-1)+offset-reference[:, k]
            if include_attitude:
                tracking_error[:, 4] = torch.atan2(torch.sin(tracking_error[:, 4]),
                                                    torch.cos(tracking_error[:, 4]))
            weight = state_weight*(2.5 if k == n-1 else 1.)
            P[:, sx, sx] += output.transpose(-1, -2) @ (weight[None, :, None]*output)
            P[:, sx, sx] += 1e-5*torch.eye(nx, device=self.model.device, dtype=self.model.dtype)
            q[:, sx] += (output.transpose(-1, -2) @ (weight*tracking_error)[..., None]).squeeze(-1)
            P[:, su, su] += torch.diag(input_weight+rate_weight)
            q[:, su] += input_weight*self.plan[:, k]
            nominal_rate = self.plan[:, k]-(self.previous if k == 0 else self.plan[:, k-1])
            q[:, su] += rate_weight*nominal_rate
            if k:
                prev = slice(ns+(k-1)*NU, ns+k*NU)
                P[:, prev, prev] += torch.diag(rate_weight)
                P[:, su, prev] -= torch.diag(rate_weight)
                P[:, prev, su] -= torch.diag(rate_weight)
                q[:, prev] -= rate_weight*nominal_rate
            bound_row = slice(ns+k*NU, ns+(k+1)*NU)
            A[:, bound_row, su] = torch.eye(NU, device=self.model.device, dtype=self.model.dtype)
            lower[:, bound_row] = self.model.lower-self.plan[:, k]
            upper[:, bound_row] = self.model.upper-self.plan[:, k]
            if self.path_constraints is not None:
                lateral_limit, depth_limit, pitch_limit_deg, slew_limit = self.path_constraints
                # Local path-normal and depth constraints on the predicted
                # next state. Reference yaw is the path tangent heading.
                yaw = reference[:, k, 4]
                normal = torch.stack((-torch.sin(yaw), torch.cos(yaw)), -1)
                nominal_lateral = (normal*(self.states[:, k+1, :2] -
                                           reference[:, k, :2])).sum(-1)
                row_base = dim+6*k
                A[:, row_base, k*nx:k*nx+2] = normal
                lower[:, row_base] = -lateral_limit-nominal_lateral
                upper[:, row_base] = lateral_limit-nominal_lateral
                A[:, row_base+1, k*nx+2] = 1.
                lower[:, row_base+1] = -depth_limit - (
                    self.states[:, k+1, 2]-reference[:, k, 2])
                upper[:, row_base+1] = depth_limit - (
                    self.states[:, k+1, 2]-reference[:, k, 2])
                A[:, row_base+2, k*nx+3] = 1.
                pitch_limit = np.deg2rad(pitch_limit_deg)
                lower[:, row_base+2] = -pitch_limit-self.states[:, k+1, 3]
                upper[:, row_base+2] = pitch_limit-self.states[:, k+1, 3]
                for axis in range(NU):
                    row = row_base+3+axis
                    A[:, row, ns+k*NU+axis] = 1.
                    if k:
                        A[:, row, ns+(k-1)*NU+axis] = -1.
                    rate = self.plan[:, k, axis] - (
                        self.previous[:, axis] if k == 0 else self.plan[:, k-1, axis])
                    lower[:, row] = -slew_limit-rate
                    upper[:, row] = slew_limit-rate
        return P, q, A, lower, upper

    def _tracking_output(self, state, include_attitude=False):
        from simulator.batch_mpc import BatchedLinearMPC
        # Reuse the exact NED velocity output Jacobian of the existing MPC.
        output, offset = BatchedLinearMPC._tracking_output(
            self, state[..., :NX], include_attitude)
        if self.actuator_aware:
            output = torch.nn.functional.pad(output, (0, self.nx-NX))
        return output, offset

    def solve(self, state, reference):
        P, q, A, lower, upper = self.build_qp(state, reference)
        if self.backend == "rti_cpu":
            import osqp
            import scipy.sparse as sparse
            solutions, valid, statuses, primal, dual = [], [], [], [], []
            for i in range(self.batch_size):
                solver = osqp.OSQP()
                solver.setup(P=sparse.csc_matrix(P[i].numpy()), q=q[i].numpy(),
                             A=sparse.csc_matrix(A[i].numpy()),
                             l=lower[i].numpy(), u=upper[i].numpy(),
                             verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                             max_iter=10000, polishing=False)
                result = solver.solve()
                ok = result.info.status_val in (1, 2) and np.isfinite(result.x).all()
                valid.append(ok)
                statuses.append(result.info.status)
                primal.append(result.info.prim_res)
                dual.append(result.info.dual_res)
                solutions.append(result.x if ok else np.zeros(P.shape[-1]))
            delta = self.model.tensor(np.stack(solutions))
            valid = torch.tensor(valid, device=self.model.device, dtype=torch.bool)
            self.last_status = statuses
            self.last_primal_residual = self.model.tensor(primal)
            self.last_dual_residual = self.model.tensor(dual)
            self.last_qp_valid = valid.clone()
        else:
            if self.solver is None:
                from simulator.rti_cuda import CudssEqualityBoxQP
                self.solver = CudssEqualityBoxQP(
                    P.shape[-1], A.shape[-2], self.horizon*self.nx, self.batch_size,
                    iterations=self.iterations, tolerance=self.tolerance)
            finite = (torch.isfinite(P).all((-1, -2)) &
                      torch.isfinite(q).all(-1) &
                      torch.isfinite(A).all((-1, -2)) &
                      torch.isfinite(lower).all(-1) &
                      torch.isfinite(upper).all(-1))
            identity = torch.eye(P.shape[-1], device=P.device, dtype=P.dtype)
            P = torch.where(finite[:, None, None], P, identity)
            q = torch.where(finite[:, None], q, torch.zeros_like(q))
            A = torch.where(finite[:, None, None], A, identity)
            lower = torch.where(finite[:, None], lower, torch.zeros_like(lower))
            upper = torch.where(finite[:, None], upper, torch.zeros_like(upper))
            try:
                delta, valid = self.solver.solve(P, q, A, lower, upper)
                valid &= finite
                self.last_error = None
            except RuntimeError as error:
                self.last_error = error
                delta = torch.zeros_like(q)
                valid = torch.zeros(self.batch_size, device=q.device,
                                    dtype=torch.bool)
        ns = self.horizon*self.nx
        new_plan = (self.plan+delta[:, ns:].reshape(self.batch_size, self.horizon, NU))
        new_plan = torch.minimum(torch.maximum(new_plan, self.model.lower), self.model.upper)
        self.plan = torch.where(valid[:, None, None], new_plan, self.plan)
        corrected_states = self.states[:, 1:]+delta[:, :ns].reshape(
            self.batch_size, self.horizon, self.nx)
        self.states[:, 1:] = torch.where(valid[:, None, None], corrected_states,
                                          self.states[:, 1:])
        self.previous = torch.where(valid[:, None], self.plan[:, 0], self.previous)
        return self.previous*self.model.control_scale, valid
