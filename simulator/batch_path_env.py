"""Batched version of ResidualUUVEnv's path tracking task.

Path geometry is loaded once from the same CSV corpus. Runtime interpolation,
progress tracking, the frozen encoder, MPC, and plant simulation run in torch.
"""

from pathlib import Path

import numpy as np
import torch

from simulator.batch_mpc import BatchedLinearMPC, BatchedUUVModel
from simulator.model import (DISTURBANCE_CORRELATION_S, DISTURBANCE_STD, DT,
                             LOOKAHEAD, MISMATCHED_PLANT, LoopPath)
from trajectories.build_agent_relative_corpus import INPUT_FEATURES
from trajectories.train_path_encoder import PathEncoder


class PathBank:
    def __init__(self, files, device):
        self.files = [Path(file) for file in files]
        if not self.files:
            raise ValueError("path bank cannot be empty")
        paths = [LoopPath(file) for file in self.files]
        count = len(paths)
        width = max(len(path.distances) for path in paths)
        distances = np.full((count, width), np.inf)
        points = np.zeros((count, width, 3))
        tangent = np.zeros_like(points)
        pitch = np.zeros((count, width))
        yaw = np.zeros((count, width))
        speed = np.zeros((count, width))
        lengths = np.zeros(count)
        durations = np.zeros(count)
        first_state = np.zeros((count, 10))
        max_window = 1
        for i, path in enumerate(paths):
            n = len(path.distances)
            distances[i, :n] = path.distances
            points[i, :n] = path.points
            tangent[i, :n] = path.tangent
            pitch[i, :n] = path.pitch
            yaw[i, :n] = path.yaw
            speed[i, :n] = path.speed
            lengths[i] = path.length
            durations[i] = path.times[-1]
            first_state[i, :3] = path.points[0]
            first_state[i, 3:6] = [path.pitch[0], path.yaw[0], path.speed[0]]
            start = np.searchsorted(path.distances,
                                    np.maximum(path.distances - 3., 0.))
            end = np.searchsorted(path.distances,
                                  np.minimum(path.distances + LOOKAHEAD,
                                             path.length), side="right")
            max_window = max(max_window, int(np.max(end-start)))
        self.device = torch.device(device)
        self.distances = torch.as_tensor(distances, device=device)
        self.points = torch.as_tensor(points, device=device)
        self.tangent = torch.as_tensor(tangent, device=device)
        self.pitch = torch.as_tensor(pitch, device=device)
        self.yaw = torch.as_tensor(yaw, device=device)
        self.speed = torch.as_tensor(speed, device=device)
        self.lengths = torch.as_tensor(lengths, device=device)
        self.durations = torch.as_tensor(durations, device=device)
        self.first_state = torch.as_tensor(first_state, device=device)
        self.window_offsets = torch.arange(max_window, device=device)

    def sample(self, path_ids, distance):
        """Interpolate position, tangent, pitch, yaw, speed at [B, K] distances."""
        distance = distance.clamp_min(0).minimum(self.lengths[path_ids, None])
        grid = self.distances[path_ids].contiguous()
        right = torch.searchsorted(grid, distance.contiguous()).clamp(1, grid.shape[1]-1)
        left = right-1
        d0 = grid.gather(1, left)
        d1 = grid.gather(1, right)
        fraction = ((distance-d0)/(d1-d0)).clamp(0, 1)
        def interp(data):
            selected = data[path_ids]
            if selected.ndim == 3:
                left_values = selected.gather(1, left[..., None].expand(-1, -1, selected.shape[-1]))
                right_values = selected.gather(1, right[..., None].expand(-1, -1, selected.shape[-1]))
                return left_values + fraction[..., None]*(right_values-left_values)
            return selected.gather(1, left) + fraction*(selected.gather(1, right)-selected.gather(1, left))
        position = interp(self.points)
        tangent = torch.nn.functional.normalize(interp(self.tangent), dim=-1)
        return position, tangent, interp(self.pitch), interp(self.yaw), interp(self.speed)

    def advance(self, path_ids, progress, position):
        """Match LoopPath.advance's local nearest-sample, monotone progress."""
        grid = self.distances[path_ids].contiguous()
        start = torch.searchsorted(grid, (progress-3.).clamp_min(0)[:, None]).squeeze(1)
        end = torch.searchsorted(grid, (progress+LOOKAHEAD).minimum(
            self.lengths[path_ids])[:, None], right=True).squeeze(1)
        indices = start[:, None] + self.window_offsets[None]
        valid = indices < end[:, None]
        safe_indices = indices.clamp_max(grid.shape[1]-1)
        points = self.points[path_ids].gather(
            1, safe_indices[..., None].expand(-1, -1, 3))
        distances = (points-position[:, None]).square().sum(-1)
        nearest = distances.masked_fill(~valid, torch.inf).argmin(-1)
        selected = safe_indices.gather(1, nearest[:, None])
        return torch.maximum(progress, grid.gather(1, selected).squeeze(1))


class BatchedPathResidualEnv:
    def __init__(self, path_bank, encoder_checkpoint, batch_size=128, device="cuda",
                 horizon=20, backend="admm", seed=20268, randomize_plant=True,
                 admm_iterations=400, attitude_pitch_weight=0.,
                 attitude_yaw_weight=0., admm_rho=2.):
        self.paths = path_bank
        self.model = BatchedUUVModel(device)
        self.device, self.dtype = self.model.device, self.model.dtype
        self.batch_size = batch_size
        self.randomize_plant = randomize_plant
        if backend in ("rti_cpu", "rti_cuda"):
            from simulator.rti_mpc import BatchedRTIMPC
            self.mpc = BatchedRTIMPC(self.model, batch_size, horizon=horizon,
                                     backend=backend, iterations=admm_iterations,
                                     attitude_pitch_weight=attitude_pitch_weight,
                                     attitude_yaw_weight=attitude_yaw_weight)
        else:
            self.mpc = BatchedLinearMPC(self.model, batch_size, horizon=horizon,
                                        backend=backend, iterations=admm_iterations,
                                        rho=admm_rho,
                                        attitude_pitch_weight=attitude_pitch_weight,
                                        attitude_yaw_weight=attitude_yaw_weight)
        if encoder_checkpoint is not None:
            checkpoint = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
            if (checkpoint["dataset"] != "agent_relative" or
                    tuple(checkpoint["input_features"]) != INPUT_FEATURES):
                raise ValueError("encoder checkpoint must use the 12x6 agent-relative features")
            self.encoder = PathEncoder(**checkpoint["config"]).to(device).eval()
            self.encoder.load_state_dict(checkpoint["state_dict"])
            self.encoder.requires_grad_(False)
            self.observation_size = checkpoint["config"]["latent_dim"] + 24
        else:
            # Subclasses can supply their own trajectory encoder and observation.
            self.encoder = None
            self.observation_size = None
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.residual_scale = self.model.tensor([150_000., np.deg2rad(15), np.deg2rad(15)])
        self.disturbance_std = self.model.tensor(DISTURBANCE_STD)
        self.decay = np.exp(-DT/DISTURBANCE_CORRELATION_S)
        self.state = torch.zeros(batch_size, 10, device=device, dtype=self.dtype)
        self.parameters = torch.ones(batch_size, 5, device=device, dtype=self.dtype)
        self.actuator = torch.zeros(batch_size, 3, device=device, dtype=self.dtype)
        self.command_history = torch.zeros(batch_size, 5, 3, device=device, dtype=self.dtype)
        self.delay_steps = torch.zeros(batch_size, 3, device=device, dtype=torch.long)
        self.tau = torch.zeros(batch_size, 3, device=device, dtype=self.dtype)
        self.disturbance = torch.zeros(batch_size, 5, device=device, dtype=self.dtype)
        self.previous_action = torch.zeros(batch_size, 3, device=device, dtype=self.dtype)
        self.base_command = torch.zeros(batch_size, 3, device=device, dtype=self.dtype)
        self.last_mpc_ok = torch.ones(batch_size, device=device, dtype=torch.bool)
        self.mpc_failure_streak = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.path_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.progress = torch.zeros(batch_size, device=device, dtype=self.dtype)
        self.steps = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.max_steps = torch.zeros_like(self.steps)
        self.finished = torch.ones(batch_size, device=device, dtype=torch.bool)

    def _uniform(self, shape, low, high):
        return low+(high-low)*torch.rand(shape, device=self.device, dtype=self.dtype,
                                         generator=self.generator)

    @torch.no_grad()
    def reset(self, path_ids, mask=None):
        if mask is None:
            mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        path_ids = torch.as_tensor(path_ids, device=self.device, dtype=torch.long)
        if path_ids.shape != (self.batch_size,):
            raise ValueError("path_ids must have shape [batch_size]")
        count = int(mask.sum().item())
        if count == 0:
            return self.observe()
        self.path_ids[mask] = path_ids[mask]
        self.state[mask] = self.paths.first_state[path_ids[mask]]
        self.progress[mask] = 0
        self.steps[mask] = 0
        self.max_steps[mask] = torch.ceil(
            1.5*self.paths.durations[path_ids[mask]]/self.mpc.dt).long()
        self.finished[mask] = False
        base = MISMATCHED_PLANT
        values = self.model.tensor([base.mass_scale, base.damping_scale,
                                    base.thrust_gain, base.elevator_gain,
                                    base.rudder_gain])
        self.parameters[mask] = values
        if self.randomize_plant:
            self.parameters[mask] *= self._uniform((count, 5), 0., 1.) * self.model.tensor(
                [.06, .14, .10, .20, .20]) + self.model.tensor([.97, .93, .95, .90, .90])
            thrust_delay = self._uniform((count,), .02, .08)
            fin_delay = self._uniform((count,), .02, .08)
            thrust_tau = self._uniform((count,), .06, .14)
            fin_tau = self._uniform((count,), .05, .12)
        else:
            thrust_delay = torch.full((count,), base.thrust_delay_s, device=self.device,
                                      dtype=self.dtype)
            fin_delay = torch.full((count,), base.fin_delay_s, device=self.device,
                                   dtype=self.dtype)
            thrust_tau = torch.full((count,), base.thrust_tau_s, device=self.device,
                                    dtype=self.dtype)
            fin_tau = torch.full((count,), base.fin_tau_s, device=self.device,
                                 dtype=self.dtype)
        self.delay_steps[mask] = torch.stack((thrust_delay, fin_delay, fin_delay), -1).div(DT).sub(1e-12).ceil().long()
        self.tau[mask] = torch.stack((thrust_tau, fin_tau, fin_tau), -1)
        self.actuator[mask] = 0
        self.command_history[mask] = 0
        self.disturbance[mask] = 0
        self.previous_action[mask] = 0
        unaffected = ~mask
        old_command = self.base_command[unaffected].clone()
        old_ok = self.last_mpc_ok[unaffected].clone()
        old_mpc = {name: getattr(self.mpc, name)[unaffected].clone()
                   for name in ("previous", "plan", "states", "last_primal_residual",
                                "last_dual_residual", "last_qp_valid")
                   if hasattr(self.mpc, name)}
        self.mpc.reset(self.state, mask)
        # ResidualUUVEnv starts its MPC at zero when actuator dynamics are present.
        self.mpc.previous[mask] = 0
        self.mpc.plan[mask] = 0
        self.base_command, self.last_mpc_ok = self._solve_mpc()
        self.base_command[unaffected] = old_command
        self.last_mpc_ok[unaffected] = old_ok
        self.mpc_failure_streak[mask] = (~self.last_mpc_ok[mask]).long()
        for name, values in old_mpc.items():
            getattr(self.mpc, name)[unaffected] = values
        return self.observe()

    def _reference(self):
        offsets = torch.arange(1, self.mpc.horizon+1, device=self.device,
                               dtype=self.dtype)*self.mpc.dt
        _, _, _, _, speed = self.paths.sample(self.path_ids, self.progress[:, None])
        distances = self.progress[:, None] + speed.clamp_min(.1)*offsets[None]
        position, tangent, _, _, target_speed = self.paths.sample(self.path_ids, distances)
        return torch.cat((position, target_speed[..., None]*tangent), -1)

    def _solve_mpc(self):
        return self.mpc.solve(self.state, self._reference())

    def _errors(self):
        position, tangent, path_pitch, path_yaw, target_speed = self.paths.sample(
            self.path_ids, self.progress[:, None])
        position, tangent = position[:, 0], tangent[:, 0]
        path_pitch, path_yaw, target_speed = path_pitch[:, 0], path_yaw[:, 0], target_speed[:, 0]
        theta, psi = self.state[:, 3], self.state[:, 4]
        ct, st, cp, sp = torch.cos(theta), torch.sin(theta), torch.cos(psi), torch.sin(psi)
        rotation = torch.stack((torch.stack((cp*ct, -sp, cp*st), -1),
                                torch.stack((sp*ct, cp, sp*st), -1),
                                torch.stack((-st, torch.zeros_like(st), ct), -1)), -2)
        position_body = torch.bmm((position-self.state[:, :3])[:, None], rotation).squeeze(1)
        position_error = self.state[:, :3]-position
        lateral = -torch.sin(path_yaw)*position_error[:, 0] + torch.cos(path_yaw)*position_error[:, 1]
        depth = position_error[:, 2]
        yaw = torch.atan2(torch.sin(path_yaw-psi), torch.cos(path_yaw-psi))
        pitch = torch.atan2(torch.sin(path_pitch-theta), torch.cos(path_pitch-theta))
        inertial_velocity = torch.bmm(rotation, self.state[:, 5:8, None]).squeeze(-1)
        speed_error = (inertial_velocity*tangent).sum(-1)-target_speed
        return position_body, lateral, depth, yaw, pitch, speed_error, rotation

    @torch.no_grad()
    def observe(self):
        offsets = self.model.tensor(4.*np.arange(12))
        requested = self.progress[:, None] + offsets[None]
        valid = requested <= self.paths.lengths[self.path_ids, None]
        points, _, path_pitch, path_yaw, _ = self.paths.sample(self.path_ids, requested)
        position_body, _, _, yaw_error, pitch_error, _, rotation = self._errors()
        body_preview = torch.bmm(points-self.state[:, None, :3], rotation)
        relative_yaw = torch.atan2(torch.sin(path_yaw-self.state[:, 4:5]),
                                   torch.cos(path_yaw-self.state[:, 4:5]))
        preview = torch.stack((body_preview[..., 0]/50, body_preview[..., 1]/50,
                               body_preview[..., 2]/50, relative_yaw/np.pi,
                               (path_pitch-self.state[:, 3:4])/.5,
                               valid.to(self.dtype)), -1).float()
        latent = self.encoder(preview)
        speed_distances = self.progress[:, None] + self.model.tensor([0., 15., 30., 45.])
        _, _, _, _, speed_preview = self.paths.sample(self.path_ids, speed_distances)
        return torch.cat((latent, position_body/10,
                          torch.stack((yaw_error, pitch_error), -1)/.3,
                          self.state[:, 5:8]/self.model.tensor([15., 5., 5.]),
                          self.state[:, 8:10]/.5, speed_preview/15,
                          self.base_command/self.model.control_scale,
                          self.actuator/self.model.control_scale,
                          self.previous_action,
                          (self.progress/self.paths.lengths[self.path_ids])[:, None]), -1).float()

    @torch.no_grad()
    def step(self, action, *, return_observation=True):
        if action.shape != (self.batch_size, 3):
            raise ValueError("action must have shape [batch_size, 3]")
        active = ~self.finished
        action = action.to(self.dtype).clamp(-1, 1)
        command = (self.base_command + action*self.residual_scale).clamp(
            min=self.model.lower*self.model.control_scale,
            max=self.model.upper*self.model.control_scale)
        change_cost = (action-self.previous_action).square().sum(-1)
        old_progress = self.progress.clone()
        tracking = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
        failed = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        track_nominal = getattr(self, "track_nominal_dynamics", False)
        if track_nominal:
            nominal_body_change = torch.zeros(self.batch_size, 5, device=self.device,
                                               dtype=self.dtype)
        track_current = getattr(self, "track_current_dynamics", False)
        if track_current:
            water_displacement = torch.zeros(self.batch_size, 3, device=self.device,
                                              dtype=self.dtype)
        for _ in range(5):
            running = active & ~failed
            noise = torch.randn(self.batch_size, 5, device=self.device, dtype=self.dtype,
                                generator=self.generator)
            next_disturbance = (self.decay*self.disturbance +
                                np.sqrt(1-self.decay**2)*self.disturbance_std*noise)
            self.disturbance = torch.where(running[:, None], next_disturbance, self.disturbance)
            delayed = self.command_history.gather(
                1, (self.delay_steps-1).clamp_min(0)[:, None]).squeeze(1)
            target = torch.where(self.delay_steps > 0, delayed, command)
            self.command_history = torch.where(running[:, None, None],
                torch.cat((command[:, None], self.command_history[:, :-1]), 1),
                self.command_history)
            alpha = torch.where(self.tau > 0, -torch.expm1(-DT/self.tau.clamp_min(1e-12)),
                                torch.ones_like(self.tau))
            next_actuator = self.actuator + alpha*(target-self.actuator)
            self.actuator = torch.where(running[:, None], next_actuator, self.actuator)
            if track_nominal:
                nominal_next = self.model.rk4(self.state, self.actuator)
                nominal_body_change += torch.where(running[:, None],
                    nominal_next[:, 5:]-self.state[:, 5:], 0.)
            if track_current:
                water_before = self.model.derivative(self.state, self.actuator)[:, :3]
            next_state = self.model.rk4(self.state, self.actuator,
                                        self.parameters, self.disturbance,
                                        current=getattr(self, "current_ned", None))
            if track_current:
                water_after = self.model.derivative(next_state, self.actuator)[:, :3]
                water_displacement += torch.where(running[:, None],
                    .5*DT*(water_before+water_after), 0.)
            self.state = torch.where(running[:, None], next_state, self.state)
            next_progress = self.paths.advance(self.path_ids, self.progress, self.state[:, :3])
            self.progress = torch.where(running, next_progress, self.progress)
            position, lateral, depth, yaw, pitch, speed, _ = self._errors()
            cost = (.4*(lateral/5).square() + .4*(depth/3).square() +
                    .2*(speed/2).square() + .1*(yaw/.3).square() +
                    .1*(pitch/.3).square())/5
            tracking += torch.where(running, cost, 0)
            failed |= running & ((position.norm(dim=-1) > 80) |
                                 (self.state[:, 3].abs() > np.deg2rad(70)) |
                                 (self.state[:, 5:8].norm(dim=-1) > 30) |
                                 ~torch.isfinite(self.state).all(-1))
        self.steps += active.long()
        endpoints = self.paths.points[self.path_ids, torch.searchsorted(
            self.paths.distances[self.path_ids].contiguous(),
            self.paths.lengths[self.path_ids, None]).squeeze(1)]
        success = active & (self.progress >= self.paths.lengths[self.path_ids]-2) & (
            (self.state[:, :3]-endpoints).norm(dim=-1) < 10)
        timeout = active & (self.steps >= self.max_steps)
        done = success | failed | timeout
        self.finished |= done
        _, _, _, _, target_speed = self.paths.sample(self.path_ids, old_progress[:, None])
        target_gain = target_speed[:, 0]*self.mpc.dt
        reward = ((self.progress-old_progress)/target_gain - tracking -
                  .01*action.square().sum(-1) - .03*change_cost +
                  100*success.to(self.dtype) - 100*(~success & (failed | timeout)).to(self.dtype))
        reward = torch.where(active, reward, 0)
        reward = torch.where(torch.isfinite(reward), reward, torch.full_like(reward, -100.))
        self.state = torch.where(torch.isfinite(self.state), self.state, torch.zeros_like(self.state))
        self.previous_action = torch.where(active[:, None], action, self.previous_action)
        old_command = self.base_command.clone()
        old_ok = self.last_mpc_ok.clone()
        old_streak = self.mpc_failure_streak.clone()
        old_primal = getattr(self.mpc, "last_primal_residual", torch.zeros_like(self.progress)).clone()
        old_dual = getattr(self.mpc, "last_dual_residual", torch.zeros_like(self.progress)).clone()
        old_qp_valid = getattr(self.mpc, "last_qp_valid", old_ok).clone()
        continuing = active & ~done
        if continuing.any():
            old_mpc = {name: getattr(self.mpc, name)[~continuing].clone()
                       for name in ("previous", "plan", "states", "last_primal_residual",
                                    "last_dual_residual", "last_qp_valid")
                       if hasattr(self.mpc, name)}
            new_command, new_ok = self._solve_mpc()
            self.base_command = torch.where(continuing[:, None], new_command, old_command)
            self.last_mpc_ok = torch.where(continuing, new_ok, old_ok)
            self.mpc_failure_streak = torch.where(
                continuing, torch.where(new_ok, 0, self.mpc_failure_streak+1),
                self.mpc_failure_streak)
            for name, values in old_mpc.items():
                getattr(self.mpc, name)[~continuing] = values
        position_body = self._errors()[0]
        info = {"success": success, "failed": failed, "timeout": timeout,
                "progress_fraction": self.progress/self.paths.lengths[self.path_ids],
                "position_error_m": position_body.norm(dim=-1),
                "mpc_ok": old_ok, "mpc_failure_streak": old_streak,
                "mpc_qp_valid": old_qp_valid,
                "mpc_primal_residual": old_primal,
                "mpc_dual_residual": old_dual, "command": command}
        if track_nominal:
            info["nominal_body_acceleration"] = nominal_body_change/self.mpc.dt
        if track_current:
            info["water_displacement_ned"] = water_displacement
        observation = self.observe() if return_observation else None
        return observation, reward.float(), done, info
