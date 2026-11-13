"""Batched 0.02 s plants, 0.1 s MPC, 1 s policy; all simulation tensors on device."""
import numpy as np
import torch

from .features import Features, wrap
from .mpc import BatchedFrequencyMPC
from .references import reference


def failure_penalty(time, duration):
    return -100*(duration-time)/duration-10


class BatchedEnv:
    def __init__(self, tasks, *, device='cuda', predictor='cusadi', iterations=400,
                 tolerance=1e-5, cuda_graph=True):
        self.tasks = tasks
        self.duration = float(tasks[0]['duration_s'])
        if any(t['duration_s'] != self.duration for t in tasks):
            raise ValueError('batch tasks must have equal duration')
        self.cycles = round(self.duration/.1)
        self.mpc = BatchedFrequencyMPC(len(tasks), device=device, predictor=predictor,
                                       iterations=iterations, tolerance=tolerance,
                                       cuda_graph=cuda_graph)
        self.d = self.mpc.dynamics
        d, b = self.d, len(tasks)
        self.combined = torch.tensor([t['plant'] == 'combined' for t in tasks], device=d.device)
        times = np.arange(self.cycles*5+401)*.02
        self.references = d.tensor(np.stack([reference(t, times) for t in tasks]))
        self.x, self.actual, self.rate, self.queue = d.initial([t['speed_mps'] for t in tasks])
        self.command_history = self.queue.clone()/d.scale
        self.last_command = self.actual/d.scale
        self.mpc.reset(self.last_command)
        self.beta = d.tensor([.5]*b)
        self.previous_target = self.beta.clone()
        self.estimated_rate = torch.zeros_like(self.actual)
        self.error_history = torch.zeros(b, 80, 3, device=d.device, dtype=d.dtype)
        self.innovation_history = torch.zeros_like(self.error_history)
        self.alive = torch.ones(b, device=d.device, dtype=torch.bool)
        self.failed = torch.zeros_like(self.alive)
        self.qp_failed = torch.zeros_like(self.alive)
        self.qp_ok = self.alive.clone()
        self.end_time = torch.zeros_like(self.beta)
        self.returns = torch.zeros_like(self.beta)
        self.squared = torch.zeros(b, 8, device=d.device, dtype=d.dtype)
        self.maximum = torch.zeros_like(self.squared)
        self.body_squared = torch.zeros(b, 4, device=d.device, dtype=d.dtype)
        self.samples = torch.zeros_like(self.beta)
        self.saturation_count = torch.zeros_like(self.beta)
        self.beta_tv = torch.zeros_like(self.beta)
        self.command_tv = torch.zeros_like(self.actual)
        self.window_names = ['low', 'high', 'transition', 'recovery']
        self.window_sums = torch.zeros(b, 4, device=d.device, dtype=d.dtype)
        self.window_counts = torch.zeros_like(self.window_sums)
        self.trace = torch.zeros(self.cycles, b, 12, device=d.device, dtype=d.dtype)
        self.last_beta_motion = torch.full_like(self.beta, -100.)
        self.k = 0
        self.features = Features(d)
        self.use_graph = cuda_graph and d.device.type == 'cuda'
        self.plant_graph = None

    def observe(self):
        return self.features.observe(self)

    def _plant_block(self, x, actual, rate, queue, command):
        states, actuators, rates, queues = [], [], [], []
        for _ in range(5):
            x, actual, rate, queue = self.d.plant_step(x, actual, rate, queue, command, self.combined)
            states.append(x)
            actuators.append(actual)
            rates.append(rate)
            queues.append(queue)
        return tuple(torch.stack(v, 1) for v in (states, actuators, rates, queues))

    def plant_block(self, command):
        inputs = (self.x, self.actual, self.rate, self.queue, command)
        if not self.use_graph:
            return self._plant_block(*inputs)
        if self.plant_graph is None:
            self.plant_inputs = tuple(v.clone() for v in inputs)
            stream = torch.cuda.Stream(device=self.d.device)
            stream.wait_stream(torch.cuda.current_stream(self.d.device))
            with torch.cuda.stream(stream):
                self._plant_block(*self.plant_inputs)
            torch.cuda.current_stream(self.d.device).wait_stream(stream)
            self.plant_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.plant_graph):
                self.plant_result = self._plant_block(*self.plant_inputs)
        for dst, src in zip(self.plant_inputs, inputs):
            dst.copy_(src)
        self.plant_graph.replay()
        return self.plant_result

    @torch.no_grad()
    def cycle(self, target):
        d, k = self.d, self.k
        live = self.alive.clone()
        prior_beta = self.beta.clone()
        self.beta = torch.where(live, self.beta+(target-self.beta).clamp(-.02, .02), self.beta)
        movement = (self.beta-prior_beta).abs()
        self.beta_tv += movement
        self.last_beta_motion = torch.where(movement > 1e-8, k*.1, self.last_beta_motion)
        state = torch.cat((self.x, self.actual/d.scale, self.command_history.flatten(1)), -1)
        refs = self.references[:, 5*(k+1):5*(k+36):5]
        command, ok = self.mpc.solve(state, refs, self.beta, live)
        normalized = command/d.scale
        # The innovation baseline is independent of the policy's chosen beta.
        baseline_beta = torch.full_like(self.beta, .5)
        if self.mpc.predictor is None:
            predicted = d.predictor_step(state, normalized, baseline_beta)
        else:
            predicted, _, _ = self.mpc.predictor.evaluate(state, normalized, baseline_beta)
        command_change = normalized-self.last_command
        # GS-21 has only box constraints. Full normalized command spans are
        # explicit reward scales, not nonexistent hard increment constraints.
        command_cost = (command_change/d.tensor([1., 2., 2.])).square().mean(-1)
        slew_cost = ((self.beta-prior_beta)/.02).square()
        self.command_tv += torch.where(live[:, None], command_change.abs(), 0.)
        saturation = ((normalized-self.mpc.lower).abs() < 1e-5) | ((normalized-self.mpc.upper).abs() < 1e-5)
        self.saturation_count += (live & ok & saturation.any(-1)).double()
        self.qp_ok = ok
        qp_failure = live & ~ok
        self.qp_failed |= qp_failure
        x, actual, rate, queue = self.plant_block(command)
        outputs = d.output(x.flatten(0, 1), actual.flatten(0, 1)).reshape(-1, 5, 8)
        ref = self.references[:, 5*k+1:5*k+6]
        errors = ref-outputs
        errors[:, :, 3:5] = wrap(errors[:, :, 3:5])
        finite = torch.isfinite(x).all(-1) & torch.isfinite(actual).all(-1)
        side = -ref[:, :, 4].sin()*errors[:, :, 0]+ref[:, :, 4].cos()*errors[:, :, 1]
        tangent = ref[:, :, 5:]/torch.linalg.vector_norm(ref[:, :, 5:], dim=-1, keepdim=True)
        along = (tangent*outputs[:, :, 5:]).sum(-1)-torch.linalg.vector_norm(ref[:, :, 5:], dim=-1)
        violation = (~finite | (side.abs() > 8) | (errors[:, :, 2].abs() > 5) |
                     (along.abs() > 2) | (outputs[:, :, 3].abs() > torch.pi*40/180))
        # Include the first finite violating sample, then stop that lane.
        bad_before = torch.cat((torch.zeros_like(violation[:, :1]), violation[:, :-1]), 1).cumsum(1) > 0
        observed = live[:, None] & ok[:, None] & ~bad_before & finite
        good = live & ok & ~violation.any(-1)
        error_safe = torch.nan_to_num(errors)
        track = (error_safe[:, :, [1, 2, 5]]/d.tensor([8., 5., 2.])).square().sum(-1)
        cost = (1e6*track+.001*slew_cost[:, None]+.001*command_cost[:, None]).clamp(max=100)
        reward = -(torch.where(observed, cost, 0.)*.02/self.duration).sum(-1)
        self.squared += torch.where(observed[:, :, None], error_safe.square(), 0.).sum(1)
        self.maximum = torch.maximum(self.maximum, torch.where(observed[:, :, None], error_safe.abs(), 0.).amax(1))
        self.body_squared += torch.where(observed[:, :, None], x[:, :, 6:10].nan_to_num().square(), 0.).sum(1)
        self.samples += observed.sum(1)
        # Failure time includes the first failing integration attempt, even for NaNs.
        first_bad = violation.long().argmax(1)
        failure_time = torch.where(qp_failure, k*.1, k*.1+(first_bad+1)*.02)
        new_failure = live & ~good
        self.end_time = torch.where(good, torch.full_like(self.beta, (k+1)*.1), self.end_time)
        self.end_time = torch.where(new_failure, failure_time, self.end_time)
        reward += torch.where(new_failure, failure_penalty(failure_time, self.duration), 0.)
        self.failed |= new_failure
        self.alive = good & (k+1 < self.cycles)
        # Preserve finite state in dead lanes so fixed-shape QP batches stay usable.
        for name, value in [('x', x[:, -1]), ('actual', actual[:, -1]),
                             ('rate', rate[:, -1]), ('queue', queue[:, -1])]:
            old = getattr(self, name)
            mask = good.reshape(-1, *([1]*(old.ndim-1)))
            setattr(self, name, torch.where(mask, value, old))
        old_actual = state[:, 10:13]
        self.estimated_rate += (1-torch.exp(d.tensor(-.1/.2)))*((self.actual/d.scale-old_actual)/.1-self.estimated_rate)
        for _ in range(5):
            self.command_history = torch.cat((normalized[:, None], self.command_history[:, :-1]), 1)
        innovation = self.x[:, [5, 8, 9]]-predicted[:, [5, 8, 9]]
        self.innovation_history = torch.cat((self.innovation_history[:, 1:], innovation[:, None]), 1)
        measured_error = (self.references[:, (k+1)*5, [1, 2, 5]]-d.output(self.x, self.actual)[:, [1, 2, 5]])/d.tensor([8., 5., 2.])
        self.error_history = torch.cat((self.error_history[:, 1:], measured_error[:, None]), 1)
        self.last_command = normalized.clone()
        t = k*.1+torch.arange(1, 6, device=d.device)*.02
        transition = ((t >= 46) & (t < 50)) | ((t >= 94) & (t < 98))
        high_middle = (t >= 50) & (t < 94)
        high = torch.stack([high_middle if task['order'] == 'low_high_low' else ~high_middle for task in self.tasks])
        recovery = ((t[None]-self.last_beta_motion[:, None] < 8) &
                    (t[None] > self.last_beta_motion[:, None]) & ~transition)
        windows = torch.stack((~high & ~transition & ~recovery,
                               high & ~transition & ~recovery,
                               transition.expand_as(observed), recovery), -1) & observed[:, :, None]
        self.window_sums += torch.where(windows, track[:, :, None], 0.).sum(1)
        self.window_counts += windows.sum(1)
        self.trace[k] = torch.cat((torch.full_like(self.beta[:, None], (k+1)*.1),
                                   self.beta[:, None], target[:, None], normalized,
                                   error_safe[:, -1, [1, 2, 5]], live[:, None].double(),
                                   self.mpc.last_primal_residual[:, None],
                                   self.mpc.last_dual_residual[:, None]), -1)
        self.k += 1
        self.returns += reward
        return reward.float()

    @torch.no_grad()
    def step(self, target):
        if self.k < 80:
            target = torch.full_like(self.beta, .5)
        self.previous_target = target.clone()
        reward = torch.zeros_like(self.beta, dtype=torch.float32)
        for _ in range(min(10, self.cycles-self.k)):
            reward += self.cycle(target)
        return reward, ~self.alive

    def summaries(self):
        result = []
        for i, task in enumerate(self.tasks):
            count = float(self.samples[i])
            rmse = (self.squared[i]/max(count, 1)).sqrt().tolist()
            result.append(dict(task_id=task['id'], failed=bool(self.failed[i]), qp_failed=bool(self.qp_failed[i]),
                               time_s=float(self.end_time[i]), reward=float(self.returns[i]),
                               J_track=sum((rmse[j]/s)**2 for j, s in zip((1, 2, 5), (8., 5., 2.))),
                               rmse=rmse, max_abs_error=self.maximum[i].tolist(),
                               body_v_w_q_r_rms=(self.body_squared[i]/max(count, 1)).sqrt().tolist(),
                               saturation_cycles=int(self.saturation_count[i]),
                               beta_total_variation=float(self.beta_tv[i]),
                               command_total_variation=self.command_tv[i].tolist(),
                               windows={name: dict(samples=int(self.window_counts[i, j]),
                                   J_track=float(self.window_sums[i, j]/self.window_counts[i, j])
                                   if self.window_counts[i, j] else None) for j, name in enumerate(self.window_names)}))
        return result
