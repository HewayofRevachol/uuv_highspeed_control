"""Independent GS-12 episodes packed into CUDA lanes."""
from __future__ import annotations

import numpy as np
import torch

from experiments.gs12_joint_route.joint_reference import joint_task_reference
from experiments.gs12_rl_beta_pilot.features import spectrum
from .mpc import BatchedGS12MPC


class BatchedBetaEnv:
    cycles_per_decision = 80
    decisions = 8

    def __init__(self, tasks, j_f75, *, device="cuda", predictor="torch",
                 admm_iterations=100, admm_rho=10., admm_tolerance=1e-3,
                 cuda_graph=True, initial_beta=.75):
        if not tasks or len(tasks) != len(j_f75):
            raise ValueError("tasks and baseline lengths must match")
        if any(float(task.get("duration_s", 64.)) != 64. for task in tasks):
            raise ValueError("GS-12 beta training requires 64-second tasks")
        if not np.isfinite(j_f75).all() or np.any(np.asarray(j_f75) < 0):
            raise ValueError("F75 baselines must be finite and nonnegative")
        if not np.isfinite(initial_beta) or not 0. <= initial_beta <= .75:
            raise ValueError("initial_beta must be finite in [0, .75]")
        self.batch_size = len(tasks)
        self.mpc = BatchedGS12MPC(self.batch_size, device=device,
                                 predictor=predictor, iterations=admm_iterations,
                                 rho=admm_rho, tolerance=admm_tolerance,
                                 cuda_graph=cuda_graph)
        self.dynamics = self.mpc.dynamics
        self.device = self.dynamics.device
        times = np.arange(round((64.+16.)/.02)+5*36+1)*.02
        grids = np.stack([joint_task_reference(task, times) for task in tasks])
        self.references = self.dynamics.tensor(grids)
        preview = np.empty((self.decisions, self.batch_size, 8), dtype=np.float64)
        for decision in range(self.decisions):
            start = 5*decision*self.cycles_per_decision
            for lane in range(self.batch_size):
                energy, rho, total_rho, low = spectrum(grids[lane, start:start+800:5])
                preview[decision, lane] = np.r_[np.clip((np.log10(np.maximum(energy, 1e-12))+12.)/12., 0., 1.),
                                                rho, total_rho, float(low)]
        self.preview = self.dynamics.tensor(preview)
        self.j_f75 = self.dynamics.tensor(j_f75)
        speeds = [task.get("speed_mps", 15.) for task in tasks]
        self.x, self.actual, self.rate, self.queue = self.dynamics.initial(speeds)
        self.mpc.reset(self.actual/self.dynamics.scale)
        self.beta = torch.full((self.batch_size,), float(initial_beta),
                               device=self.device, dtype=torch.float64)
        self.previous_target = self.beta.clone()
        self.previous_j = torch.zeros_like(self.beta)
        self.previous_slew = torch.zeros_like(self.beta)
        self.alive = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        self.failed = torch.zeros_like(self.alive)
        self.qp_failed = torch.zeros_like(self.alive)
        self.violated = torch.zeros_like(self.alive)
        self.saturated = torch.zeros_like(self.alive)
        self.k = 0
        self.decision = 0
        self.window_sum = torch.zeros_like(self.beta)
        self.window_count = torch.zeros_like(self.beta)
        self.last_command = self.actual.clone()
        self.end_cycle = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)

    @torch.no_grad()
    def observe(self):
        if self.decision >= self.decisions:
            raise RuntimeError("episode batch complete")
        ref = self.references[:, 5*self.k]
        measured = self.dynamics.output(self.x, self.actual)
        error = torch.stack(((ref[:, 1]-measured[:, 1])/8.,
                             (ref[:, 2]-measured[:, 2])/5.,
                             (ref[:, 5]-measured[:, 5])/2.), -1).clamp(-2., 2.)
        values = torch.cat((self.preview[self.decision], error,
                            self.actual/self.dynamics.scale,
                            self.beta[:, None], self.previous_target[:, None],
                            torch.full_like(self.beta[:, None], self.k*.1/64.),
                            torch.log1p(self.previous_j/1e-9)[:, None].clamp(0., 20.),
                            self.previous_slew[:, None].clamp(0., 1.)), -1)
        if values.shape != (self.batch_size, 19):
            raise RuntimeError("invalid observation shape")
        return values.to(torch.float32)

    @torch.no_grad()
    def advance_control_cycle(self, target, segment_error, segment_count, segment_slew):
        if self.k >= 640:
            raise RuntimeError("batch finished")
        active = self.alive.clone()
        self.beta = torch.where(active, (self.beta+(target-self.beta).clamp(-.02, .02)).clamp(0., .75),
                                self.beta)
        state = torch.cat((self.x, self.actual/self.dynamics.scale,
                           (self.queue/self.dynamics.scale).reshape(self.batch_size, 9)), -1)
        refs = self.references[:, 5*(self.k+1):5*(self.k+36):5]
        command, qp_ok = self.mpc.solve(state, refs, self.beta, active)
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
            finite = (torch.isfinite(x_new).all(-1) & torch.isfinite(actual_new).all(-1))
            self.x = torch.where((live & finite)[:, None], x_new, self.x)
            self.actual = torch.where((live & finite)[:, None], actual_new, self.actual)
            self.rate = torch.where((live & finite)[:, None], rate_new, self.rate)
            self.queue = torch.where((live & finite)[:, None, None], queue_new, self.queue)
            ref = self.references[:, 5*self.k+j+1]
            output = self.dynamics.output(self.x, self.actual)
            difference = ((ref[:, [1, 2, 5]]-output[:, [1, 2, 5]]) /
                          self.dynamics.tensor([8., 5., 2.]))
            squared = difference.square().sum(-1)
            segment_error += torch.where(live & finite, squared, torch.zeros_like(squared))
            segment_count += (live & finite).to(segment_count.dtype)
            now = (5*self.k+j+1)*.02
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

    @torch.no_grad()
    def step_target(self, target):
        if self.decision >= self.decisions or target.shape != (self.batch_size,):
            raise ValueError("invalid target batch")
        if not bool(torch.isfinite(target).all()) or bool(((target < 0.) | (target > .75)).any()):
            raise ValueError("beta target must be finite in [0, .75]")
        started = self.alive.clone()
        error = torch.zeros_like(self.beta)
        count = torch.zeros_like(self.beta)
        slew = torch.zeros_like(self.beta)
        previous = self.previous_target.clone()
        for _ in range(self.cycles_per_decision):
            if not bool(self.alive.any()):
                break
            error, count, slew = self.advance_control_cycle(target, error, count, slew)
        j_segment = error/count.clamp_min(1.)
        self.decision += 1
        self.previous_j = torch.where(started, j_segment, self.previous_j)
        self.previous_slew = torch.where(started, slew, self.previous_slew)
        self.previous_target = torch.where(started, target, previous)
        reward = -(j_segment/self.j_f75.clamp_min(1e-9)).clamp(max=20.)
        reward -= .02*(target-previous).square()
        reward -= self.failed.to(reward.dtype)*20.*(9-self.decision)
        reward = torch.where(started, reward, torch.zeros_like(reward))
        done = ~self.alive
        return reward.to(torch.float32), done, {
            "J_seg": j_segment, "beta_target": target,
            "failed": self.failed.clone(), "qp_failed": self.qp_failed.clone(),
            "violated": self.violated.clone()}

    @torch.no_grad()
    def summary(self):
        score = self.window_sum/self.window_count.clamp_min(1.)
        return [{"task_id": None, "J_track": float(score[i]) if
                 self.window_count[i] > 0 and not self.failed[i] else None,
                 "status": ("qp_failure" if self.qp_failed[i] else
                            "early_violation" if self.violated[i] else
                            "complete_constrained" if self.saturated[i] else "complete_valid"),
                 "control_cycles": int(self.end_cycle[i])}
                for i in range(self.batch_size)]
