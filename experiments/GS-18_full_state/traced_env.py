"""GS-12 GPU closed loop with a trajectory recorder at every plant substep."""
from __future__ import annotations

import torch

from experiments.gs12_rl_beta_gpu_admm.environment import BatchedBetaEnv


class TracedBetaEnv(BatchedBetaEnv):
    def __init__(self, tasks, j_f75, **kwargs):
        super().__init__(tasks, j_f75, **kwargs)
        b, device = self.batch_size, self.device
        self.trace_x = torch.zeros((b, 3201, 10), dtype=torch.float64, device=device)
        self.trace_actual = torch.zeros((b, 3201, 3), dtype=torch.float64, device=device)
        self.trace_valid = torch.zeros((b, 3201), dtype=torch.bool, device=device)
        self.trace_command = torch.zeros((b, 640, 3), dtype=torch.float64, device=device)
        self.trace_beta = torch.zeros((b, 640), dtype=torch.float64, device=device)
        self.trace_qp_valid = torch.zeros((b, 640), dtype=torch.bool, device=device)
        self.trace_qp_residual = torch.zeros((b, 640, 2), dtype=torch.float32, device=device)
        self.trace_x[:, 0] = self.x
        self.trace_actual[:, 0] = self.actual
        self.trace_valid[:, 0] = True

    @torch.no_grad()
    def advance_control_cycle(self, target, segment_error, segment_count, segment_slew):
        # Keep the GS-12 control and failure semantics; add only trace writes.
        active = self.alive.clone()
        self.beta = torch.where(active, (self.beta+(target-self.beta).clamp(-.02, .02)).clamp(0., .75),
                                self.beta)
        state = torch.cat((self.x, self.actual/self.dynamics.scale,
                           (self.queue/self.dynamics.scale).reshape(self.batch_size, 9)), -1)
        refs = self.references[:, 5*(self.k+1):5*(self.k+36):5]
        command, qp_ok = self.mpc.solve(state, refs, self.beta, active)
        self.trace_command[:, self.k] = command
        self.trace_beta[:, self.k] = self.beta
        self.trace_qp_valid[:, self.k] = qp_ok
        self.trace_qp_residual[:, self.k, 0] = self.mpc.last_primal_residual
        self.trace_qp_residual[:, self.k, 1] = self.mpc.last_dual_residual
        if self.k:
            segment_slew = torch.maximum(segment_slew,
                torch.where(active, ((command-self.last_command)/self.dynamics.scale).abs().amax(-1),
                            torch.zeros_like(segment_slew)))
        self.last_command = torch.where(active[:, None], command, self.last_command)
        lower = self.dynamics.tensor([0., -1., -1.])
        normalized = command/self.dynamics.scale
        self.saturated |= active & ((normalized >= 1.-1e-6) |
                                     (normalized <= lower+1e-6)).any(-1)
        for j in range(5):
            if not bool(self.alive.any()):
                break
            live = self.alive.clone()
            x_new, actual_new, rate_new, queue_new = self.dynamics.plant_step(
                self.x, self.actual, self.rate, self.queue, command)
            finite = torch.isfinite(x_new).all(-1) & torch.isfinite(actual_new).all(-1)
            self.x = torch.where((live & finite)[:, None], x_new, self.x)
            self.actual = torch.where((live & finite)[:, None], actual_new, self.actual)
            self.rate = torch.where((live & finite)[:, None], rate_new, self.rate)
            self.queue = torch.where((live & finite)[:, None, None], queue_new, self.queue)
            index = 5*self.k+j+1
            self.trace_x[:, index] = self.x
            self.trace_actual[:, index] = self.actual
            self.trace_valid[:, index] = live & finite
            ref = self.references[:, index]
            output = self.dynamics.output(self.x, self.actual)
            difference = ((ref[:, [1, 2, 5]]-output[:, [1, 2, 5]]) /
                          self.dynamics.tensor([8., 5., 2.]))
            squared = difference.square().sum(-1)
            segment_error += torch.where(live & finite, squared, torch.zeros_like(squared))
            segment_count += (live & finite).to(segment_count.dtype)
            now = index*.02
            if 32. <= now <= 64.:
                self.window_sum += torch.where(live & finite, squared, torch.zeros_like(squared))
                self.window_count += (live & finite).to(self.window_count.dtype)
            pos_error = output[:, :3]-ref[:, :3]
            side = -torch.sin(ref[:, 4])*pos_error[:, 0]+torch.cos(ref[:, 4])*pos_error[:, 1]
            tangent = ref[:, 5:]/torch.linalg.vector_norm(ref[:, 5:], dim=-1)[:, None]
            speed_error = (tangent*output[:, 5:]).sum(-1)-torch.linalg.vector_norm(ref[:, 5:], dim=-1)
            violation = ((side.abs() > 8.) | (pos_error[:, 2].abs() > 5.) |
                         (speed_error.abs() > 2.) | (output[:, 3].abs() > torch.pi*40./180.) |
                         ~finite)
            new_failure = live & violation
            self.violated |= new_failure
            self.failed |= new_failure
            self.alive &= ~new_failure
        new_qp_failure = active & ~qp_ok
        self.qp_failed |= new_qp_failure
        self.failed |= new_qp_failure
        self.alive &= ~new_qp_failure
        self.k += 1
        if self.k == 640:
            self.alive.zero_()
        self.end_cycle = torch.where(active & ~self.alive,
                                     torch.full_like(self.end_cycle, self.k), self.end_cycle)
        return segment_error, segment_count, segment_slew
