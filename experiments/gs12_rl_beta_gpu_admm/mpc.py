"""Batched GS-12 condensed RTI QP with CUDA box ADMM."""
from __future__ import annotations

import torch
from torch.func import jacfwd, vmap

from simulator.batch_mpc import BatchedLinearMPC, BatchedUUVModel
from .dynamics import Dynamics


class BoxADMM:
    def __init__(self, batch_size, variables=105, iterations=400, rho=10., tolerance=1e-3,
                 cuda_graph=True):
        if iterations < 1 or rho <= 0 or tolerance <= 0:
            raise ValueError("invalid ADMM settings")
        self.batch_size, self.variables = batch_size, variables
        self.iterations, self.rho, self.tolerance = iterations, rho, tolerance
        self.cuda_graph = cuda_graph
        self.graph = None

    def _iterate(self, factor, gradient, lower, upper):
        z = torch.zeros_like(gradient).clamp(lower, upper)
        y = torch.zeros_like(z)
        for _ in range(self.iterations):
            x = torch.cholesky_solve(
                (-gradient+self.rho*(z-y))[..., None], factor).squeeze(-1)
            old = z
            z = (x+y).clamp(lower, upper)
            y = y+x-z
        primal = (x-z).abs().amax(-1)
        dual = (self.rho*(z-old)).abs().amax(-1)
        return z, primal, dual

    def solve(self, hessian, gradient, lower, upper):
        b, n = gradient.shape
        if (b, n) != (self.batch_size, self.variables):
            raise ValueError("QP batch shape changed")
        finite = (torch.isfinite(hessian).all((-1, -2)) &
                  torch.isfinite(gradient).all(-1) &
                  torch.isfinite(lower).all(-1) & torch.isfinite(upper).all(-1))
        identity = torch.eye(n, device=gradient.device, dtype=gradient.dtype)
        matrix = torch.where(finite[:, None, None], hessian, identity) + self.rho*identity
        factor, info = torch.linalg.cholesky_ex(matrix)
        good = finite & (info == 0)
        factor = torch.where(good[:, None, None], factor, identity)
        gradient = torch.where(good[:, None], gradient, torch.zeros_like(gradient))
        lower = torch.where(good[:, None], lower, torch.zeros_like(lower))
        upper = torch.where(good[:, None], upper, torch.zeros_like(upper))
        if self.cuda_graph and gradient.is_cuda:
            if self.graph is None:
                self.static = tuple(x.clone() for x in (factor, gradient, lower, upper))
                stream = torch.cuda.Stream(device=gradient.device)
                stream.wait_stream(torch.cuda.current_stream(gradient.device))
                with torch.cuda.stream(stream):
                    self._iterate(*self.static)
                torch.cuda.current_stream(gradient.device).wait_stream(stream)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.result = self._iterate(*self.static)
            for dst, src in zip(self.static, (factor, gradient, lower, upper)):
                dst.copy_(src)
            self.graph.replay()
            solution, primal, dual = self.result
        else:
            solution, primal, dual = self._iterate(factor, gradient, lower, upper)
        valid = (good & torch.isfinite(solution).all(-1) &
                 (primal <= self.tolerance) & (dual <= self.tolerance))
        return solution, valid, primal, dual


class BatchedGS12MPC:
    def __init__(self, batch_size, device="cuda", horizon=35, iterations=100,
                 rho=10., tolerance=1e-3, cuda_graph=True, predictor="torch"):
        if horizon != 35:
            raise ValueError("this experiment freezes horizon=35")
        self.batch_size, self.horizon = batch_size, horizon
        self.dynamics = Dynamics(device)
        self.model = BatchedUUVModel(device)
        self.lower = self.dynamics.tensor([0., -1., -1.])
        self.upper = self.dynamics.tensor([1., 1., 1.])
        self.admm = BoxADMM(batch_size, horizon*3, iterations, rho,
                            tolerance, cuda_graph)
        self.retry_admm = BoxADMM(batch_size, horizon*3, max(iterations, 400),
                                  rho, tolerance, cuda_graph)
        dim = horizon*3
        self.eye = torch.eye(dim, device=self.dynamics.device, dtype=torch.float32)
        self.input_weight = self.dynamics.tensor([.15, .08, .08]).float().repeat(horizon)
        self.rate_weight = self.dynamics.tensor([1.8, .245, .245]).float().repeat(horizon)
        self.difference = self.eye.clone()
        for k in range(1, horizon):
            self.difference[3*k:3*k+3, 3*(k-1):3*k] = -torch.eye(
                3, device=self.dynamics.device, dtype=torch.float32)
        self.state_weight = self.dynamics.tensor([3., 3., 4.2, 100., 30., 5., 5., 5.]).float()
        self.lower_qp = self.lower.float()
        self.upper_qp = self.upper.float()
        if predictor == "cusadi":
            from .cusadi22 import CuSADi22
            self.predictor = CuSADi22(batch_size, device)
            self.jacobian = None
        elif predictor == "torch":
            self.predictor = None
            self.jacobian = vmap(jacfwd(self.dynamics.predictor_step, argnums=(0, 1)))
        else:
            raise ValueError("predictor must be torch or cusadi")
        self.builder_graph = None
        self.use_builder_graph = cuda_graph and predictor == "cusadi"
        self.plan = None
        self.reset(self.dynamics.tensor([.6, 0., 0.]).expand(batch_size, 3))

    def reset(self, actual_normalized, mask=None):
        if actual_normalized.shape != (self.batch_size, 3):
            raise ValueError("expected [batch, 3] actual commands")
        if self.plan is None:
            self.previous = actual_normalized.clone()
            self.plan = actual_normalized[:, None].expand(-1, self.horizon, -1).clone()
        else:
            if mask is None:
                mask = torch.ones(self.batch_size, device=self.plan.device, dtype=torch.bool)
            self.previous = torch.where(mask[:, None], actual_normalized, self.previous)
            self.plan = torch.where(mask[:, None, None],
                                    actual_normalized[:, None], self.plan)

    def _tracking(self, state):
        matrix, offset = BatchedLinearMPC._tracking_output(self, state[:, :10], True)
        return torch.nn.functional.pad(matrix, (0, 12)), offset

    def _build_qp(self, state, reference, beta, plan, previous):
        b, n = self.batch_size, self.horizon
        dim = 3*n
        trajectory = state
        influence = torch.zeros(b, 22, dim, device=state.device, dtype=torch.float32)
        hessian = 1e-6*self.eye.expand(b, -1, -1).clone()
        gradient = torch.zeros(b, dim, device=state.device, dtype=torch.float32)
        flat_plan = plan.reshape(b, dim).float()
        rate_nominal = flat_plan @ self.difference.T
        rate_nominal[:, :3] -= previous.float()
        hessian += torch.diag(self.input_weight) + self.difference.T @ (
            self.rate_weight[:, None]*self.difference)
        gradient += self.input_weight*flat_plan + (self.rate_weight*rate_nominal) @ self.difference
        for k in range(n):
            nominal = plan[:, k]
            if self.predictor is None:
                a, u = self.jacobian(trajectory, nominal, beta)
                trajectory = self.dynamics.predictor_step(trajectory, nominal, beta)
            else:
                trajectory, a, u = self.predictor.evaluate(trajectory, nominal, beta)
            influence = a.float() @ influence
            influence[:, :, 3*k:3*k+3] += u.float()
            output, offset = self._tracking(trajectory)
            error = ((output @ trajectory[..., None]).squeeze(-1)+offset-reference[:, k]).float()
            error[:, 4] = torch.atan2(torch.sin(error[:, 4]), torch.cos(error[:, 4]))
            sensitivity = output.float() @ influence
            weight = self.state_weight*(2.5 if k == n-1 else 1.)
            hessian += sensitivity.transpose(-1, -2) @ (weight[None, :, None]*sensitivity)
            gradient += (sensitivity.transpose(-1, -2) @
                         (weight*error)[..., None]).squeeze(-1)
        lower = self.lower_qp.repeat(n)-flat_plan
        upper = self.upper_qp.repeat(n)-flat_plan
        return hessian, gradient, lower, upper

    @torch.no_grad()
    def solve(self, state, reference, beta, active=None):
        b, n = self.batch_size, self.horizon
        if state.shape != (b, 22) or reference.shape != (b, n, 8) or beta.shape != (b,):
            raise ValueError("invalid GS-12 MPC input shapes")
        if active is None:
            active = torch.ones(b, device=state.device, dtype=torch.bool)
        plan = torch.cat((self.plan[:, 1:], self.plan[:, -1:]), 1)
        previous = self.previous
        if self.use_builder_graph:
            if self.builder_graph is None:
                self.builder_inputs = tuple(x.clone() for x in
                                            (state, reference, beta, plan, previous))
                stream = torch.cuda.Stream(device=state.device)
                stream.wait_stream(torch.cuda.current_stream(state.device))
                with torch.cuda.stream(stream):
                    self._build_qp(*self.builder_inputs)
                torch.cuda.current_stream(state.device).wait_stream(stream)
                self.builder_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.builder_graph):
                    self.builder_outputs = self._build_qp(*self.builder_inputs)
            for dst, src in zip(self.builder_inputs, (state, reference, beta, plan, previous)):
                dst.copy_(src)
            self.builder_graph.replay()
            hessian, gradient, lower, upper = self.builder_outputs
        else:
            hessian, gradient, lower, upper = self._build_qp(
                state, reference, beta, plan, previous)
        qp_dtype = torch.float32 if state.is_cuda else state.dtype
        qp = tuple(value.to(qp_dtype) for value in
                   (hessian, gradient, lower, upper))
        delta, valid, primal, dual = self.admm.solve(*qp)
        if bool((active & ~valid).any()) and self.retry_admm.iterations > self.admm.iterations:
            retry_delta, retry_valid, retry_primal, retry_dual = self.retry_admm.solve(*qp)
            delta = torch.where(retry_valid[:, None], retry_delta, delta)
            primal = torch.where(retry_valid, retry_primal, primal)
            dual = torch.where(retry_valid, retry_dual, dual)
            valid |= retry_valid
        delta = delta.to(state.dtype)
        valid = valid & active
        candidate = (plan+delta.reshape(b, n, 3)).clamp(self.lower, self.upper)
        self.plan = torch.where(valid[:, None, None], candidate, plan)
        self.previous = torch.where(valid[:, None], self.plan[:, 0], previous)
        self.last_qp_valid = valid
        self.last_primal_residual = primal
        self.last_dual_residual = dual
        return self.previous*self.model.control_scale, valid
