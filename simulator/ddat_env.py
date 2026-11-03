"""DDAT control interface on the existing batched UUV path task.

The policy sees a body-frame path preview, tracking error, state, a filtered
model-error estimate and an estimated steady current. Its three outputs
command thrust and pitch/yaw rates; nominal control maps them to actuators.
"""

import numpy as np
import torch

from simulator.batch_path_env import BatchedPathResidualEnv
from simulator.fsp_predictor import FilteredActuationPredictor


class DDATPathEnv(BatchedPathResidualEnv):
    preview_size = 12 * 6
    # Append solver diagnostics after the original 35 auxiliary features so
    # older 107-input actors can still use the leading observation columns.
    observation_size = preview_size + 35 + 5

    def __init__(self, path_bank, batch_size=128, device="cpu", horizon=30,
                 backend="admm", seed=1026, randomize_plant=True,
                 admm_iterations=400, adaptation_time=.5,
                 current_max_mps=1., current_filter_time=1.,
                 current_profile="steady",
                 attitude_pitch_weight=100., attitude_yaw_weight=30.,
                 fsp_enabled=False, fsp_filter_time=.4, fsp_preview_time=None,
                 admm_rho=2., model_factory=None):
        if (adaptation_time <= 0 or current_filter_time <= 0 or
                current_max_mps < 0 or not np.isfinite(current_max_mps)):
            raise ValueError("adaptation/filter times must be positive and current_max_mps finite, nonnegative")
        if current_profile not in ("steady", "harmonic"):
            raise ValueError("current_profile must be steady or harmonic")
        super().__init__(path_bank, None, batch_size, device, horizon, backend,
                         seed, randomize_plant, admm_iterations,
                         admm_rho=admm_rho,
                         attitude_pitch_weight=attitude_pitch_weight,
                         attitude_yaw_weight=attitude_yaw_weight,
                         model_factory=model_factory)
        self.observation_size = type(self).observation_size
        self.disturbance_estimate = torch.zeros(batch_size, 5, device=self.device,
                                                dtype=self.dtype)
        # True NED current belongs only to the plant; the policy receives its
        # filtered kinematic estimate, expressed in the body frame.
        self.current_ned = torch.zeros(batch_size, 3, device=self.device,
                                       dtype=self.dtype)
        self.current_estimate = torch.zeros_like(self.current_ned)
        self.current_max_mps = current_max_mps
        self.current_profile = current_profile
        # Keep wave sampling separate from plant/noise sampling so paired seeds
        # use the same plant and body-acceleration disturbance stream.
        self.current_generator = torch.Generator(device=self.device).manual_seed(
            int(seed) + 730101)
        self.current_base = torch.zeros_like(self.current_ned)
        self.current_phase = torch.zeros(batch_size, 2, device=self.device,
                                         dtype=self.dtype)
        self.current_wave = torch.zeros(batch_size, 2, 3, device=self.device,
                                        dtype=self.dtype)
        self.current_frequency_hz = self.model.tensor([.18, .55])
        self.current_scale = max(1., current_max_mps)
        self.current_gain = 1. - np.exp(-self.mpc.dt/current_filter_time)
        self.adaptation_gain = 1. - np.exp(-self.mpc.dt/adaptation_time)
        self.fsp = (FilteredActuationPredictor(
            self.model, batch_size, control_dt=self.mpc.dt,
            filter_time=fsp_filter_time,
            preview_time=fsp_preview_time) if fsp_enabled else None)
        self.estimate_limit = self.model.tensor([2., 2., 2., .2, .2])
        self.track_nominal_dynamics = True
        self.track_current_dynamics = True
        # This scale represents the entire actuator range when converting the
        # low-level controller's physical command to the parent plant API.
        self.residual_scale = 2*self.model.control_scale

    @torch.no_grad()
    def reset(self, path_ids, mask=None):
        if mask is None:
            mask = torch.ones(self.batch_size, device=self.device, dtype=torch.bool)
        if self.fsp is not None:
            self.fsp.reset(mask)
            self.base_command[mask] = 0
        self.disturbance_estimate[mask] = 0
        self.current_estimate[mask] = 0
        count = int(mask.sum().item())
        if count:
            speed = self._uniform((count,), 0., self.current_max_mps)
            bearing = self._uniform((count,), -np.pi, np.pi)
            self.current_ned[mask] = torch.stack((speed*torch.cos(bearing),
                speed*torch.sin(bearing), torch.zeros_like(speed)), -1)
            self.current_base[mask] = self.current_ned[mask]
            if self.current_profile == "harmonic":
                # Base + two bounded, coherent horizontal components. Their
                # magnitudes sum to at most current_max_mps by construction.
                self.current_base[mask] *= .35
                phase = 2*np.pi*torch.rand((count, 2), device=self.device,
                    dtype=self.dtype, generator=self.current_generator)
                direction = 2*np.pi*torch.rand((count, 2), device=self.device,
                    dtype=self.dtype, generator=self.current_generator)
                amplitude = self.current_max_mps*(.15 + .10*torch.rand(
                    (count, 2), device=self.device, dtype=self.dtype,
                    generator=self.current_generator))
                self.current_phase[mask] = phase
                self.current_wave[mask] = torch.stack((
                    amplitude*torch.cos(direction), amplitude*torch.sin(direction),
                    torch.zeros_like(amplitude)), -1)
                self.current_ned[mask] = self.current_base[mask] + (
                    self.current_wave[mask]*phase.cos()[..., None]).sum(1)
        return super().reset(path_ids, mask)

    @torch.no_grad()
    def _advance_current(self, dt, active):
        if self.current_profile == "steady" or not active.any():
            return
        phase = self.current_phase + 2*np.pi*dt*self.current_frequency_hz
        self.current_phase = torch.where(active[:, None], phase,
                                         self.current_phase)
        value = self.current_base + (
            self.current_wave*self.current_phase.cos()[..., None]).sum(1)
        self.current_ned = torch.where(active[:, None], value,
                                       self.current_ned)

    def _reference(self, time_offset=0.):
        """Give MPC path position, attitude and drift-corrected NED velocity."""
        offsets = torch.arange(1, self.mpc.horizon+1, device=self.device,
                               dtype=self.dtype)*self.mpc.dt
        _, _, _, _, speed = self.paths.sample(self.path_ids, self.progress[:, None])
        distances = self.progress[:, None] + speed.clamp_min(.1)*(offsets[None]+time_offset)
        position, tangent, pitch, yaw, target_speed = self.paths.sample(
            self.path_ids, distances)
        velocity = target_speed[..., None]*tangent
        reference = torch.cat((position, pitch[..., None], yaw[..., None], velocity), -1)
        reference[..., :3] -= offsets[None, :, None]*self.current_estimate[:, None]
        reference[..., 5:8] -= self.current_estimate[:, None]
        return reference

    def _solve_mpc(self):
        if self.fsp is None:
            return super()._solve_mpc()
        self.fsp.observe(self.state)
        preview = self.fsp.preview(
            self.state, self.actuator, self.command_history,
            self.base_command, self.current_estimate)
        return self.mpc.solve(preview, self._reference(self.fsp.preview_time))

    @torch.no_grad()
    def observe(self):
        offsets = self.model.tensor(4.*np.arange(12))
        requested = self.progress[:, None] + offsets[None]
        valid = requested <= self.paths.lengths[self.path_ids, None]
        points, _, path_pitch, path_yaw, _ = self.paths.sample(self.path_ids, requested)
        position_body, _, _, yaw_error, pitch_error, speed_error, rotation = self._errors()
        body_preview = torch.bmm(points-self.state[:, None, :3], rotation)
        relative_yaw = torch.atan2(torch.sin(path_yaw-self.state[:, 4:5]),
                                   torch.cos(path_yaw-self.state[:, 4:5]))
        preview = torch.stack((body_preview[..., 0]/50, body_preview[..., 1]/50,
                               body_preview[..., 2]/50, relative_yaw/np.pi,
                               (path_pitch-self.state[:, 3:4])/.5,
                               valid.to(self.dtype)), -1).flatten(1)
        speed_distances = self.progress[:, None] + self.model.tensor([0., 15., 30., 45.])
        _, _, _, _, speed_preview = self.paths.sample(self.path_ids, speed_distances)
        body_state = torch.cat((self.state[:, 3:5]/self.model.tensor([.5, np.pi]),
                                self.state[:, 5:8]/self.model.tensor([15., 5., 5.]),
                                self.state[:, 8:10]/.5), -1)
        error = torch.cat((position_body/10,
                           torch.stack((pitch_error, yaw_error), -1)/.3,
                           speed_error[:, None]/5), -1)
        current_body = torch.bmm(self.current_estimate[:, None], rotation).squeeze(1)
        residual_scale = (1e-7 if self.mpc.backend == "osqp" else
                          getattr(self.mpc, "tolerance", 1e-3))
        def residual_feature(name):
            residual = getattr(self.mpc, name, torch.zeros_like(self.progress))
            residual = torch.where(torch.isfinite(residual), residual,
                                   torch.full_like(residual, float("inf")))
            return torch.log1p((residual.clamp_min(0)/residual_scale).clamp_max(
                np.expm1(10.)))/10
        # Solver features: success, QP validity, failed-solve streak (20-step
        # cap), and log-scaled primal/dual residuals relative to tolerance.
        solver = torch.stack((
            self.last_mpc_ok.to(self.dtype),
            getattr(self.mpc, "last_qp_valid", self.last_mpc_ok).to(self.dtype),
            self.mpc_failure_streak.clamp_max(20).to(self.dtype)/20,
            residual_feature("last_primal_residual"),
            residual_feature("last_dual_residual")), -1)
        aux = torch.cat((body_state, error,
                         self.disturbance_estimate/self.estimate_limit,
                         current_body/self.current_scale,
                         speed_preview/15,
                         self.base_command/self.model.control_scale,
                         self.actuator/self.model.control_scale,
                         self.previous_action,
                         (self.progress/self.paths.lengths[self.path_ids])[:, None],
                         solver), -1)
        return torch.cat((preview, aux), -1).float()

    @torch.no_grad()
    def _low_level_command(self, action):
        """Map thrust and q/r setpoints to thrust/elevator/rudder commands."""
        base = self.base_command
        state = self.state
        nominal = self.model.derivative(state, base)
        desired_thrust = base[:, 0] + 150_000.*action[:, 0]
        # Nominal next-step angular rates plus a policy correction in rad/s.
        desired_rates = state[:, 8:10] + self.mpc.dt*nominal[:, 8:10] + .03*action[:, 1:]
        desired_acceleration = (desired_rates-state[:, 8:10])/self.mpc.dt
        desired_acceleration -= self.disturbance_estimate[:, 3:5]
        gain_columns = []
        for axis in (1, 2):
            offset = torch.zeros_like(base)
            offset[:, axis] = 1.
            gain_columns.append((self.model.derivative(state, base+offset)-nominal)[:, 8:10])
        gain = torch.stack(gain_columns, -1)
        # Damped least squares stays finite when fin authority vanishes at low speed.
        regularizer = 1e-4*torch.eye(2, device=self.device, dtype=self.dtype)
        correction = torch.linalg.solve(
            gain.transpose(-1, -2) @ gain + regularizer,
            (gain.transpose(-1, -2) @
             (desired_acceleration-nominal[:, 8:10])[..., None])).squeeze(-1)
        surge_gain = (self.model.derivative(
            state, base+self.model.tensor([1., 0., 0.]))-nominal)[:, 5]
        thrust_compensation = -self.disturbance_estimate[:, 0]/surge_gain.clamp_min(1e-8)
        command = torch.cat(((desired_thrust+thrust_compensation)[:, None],
                             base[:, 1:]+correction), -1)
        command = command.clamp(min=self.model.lower*self.model.control_scale,
                                max=self.model.upper*self.model.control_scale)
        return command, desired_rates

    @torch.no_grad()
    def step(self, action):
        if action.shape != (self.batch_size, 3):
            raise ValueError("action must have shape [batch_size, 3]")
        active = ~self.finished
        old_state = self.state.clone()
        action = action.to(self.dtype).clamp(-1., 1.)
        command, desired_rates = self._low_level_command(action)
        plant_action = ((command-self.base_command)/self.residual_scale).clamp(-1., 1.)
        if self.fsp is not None:
            self.fsp.expect_next(old_state, self.actuator, self.command_history,
                                 command, self.current_estimate, active)
        # The current and disturbance estimates change below, so the parent's
        # observation would be discarded before we build the final one.
        _, reward, done, info = super().step(plant_action, return_observation=False)
        measured_acceleration = (self.state[:, 5:]-old_state[:, 5:])/self.mpc.dt
        innovation = (measured_acceleration-info["nominal_body_acceleration"]).clamp(
            min=-self.estimate_limit, max=self.estimate_limit)
        finite = torch.isfinite(innovation).all(-1)
        updated = ((1-self.adaptation_gain)*self.disturbance_estimate +
                   self.adaptation_gain*innovation)
        self.disturbance_estimate = torch.where(
            (active & finite & ~info["failed"])[:, None], updated,
                                                 self.disturbance_estimate)
        current_innovation = ((self.state[:, :3]-old_state[:, :3]-
                               info["water_displacement_ned"])/self.mpc.dt)
        current_innovation = current_innovation.clamp(-2*self.current_scale,
                                                      2*self.current_scale)
        current_updated = ((1-self.current_gain)*self.current_estimate +
                           self.current_gain*current_innovation)
        current_valid = torch.isfinite(current_innovation).all(-1)
        self.current_estimate = torch.where(
            (active & current_valid & ~info["failed"])[:, None],
            current_updated, self.current_estimate)
        info["desired_thrust_and_rates"] = torch.cat((command[:, :1],
                                                       desired_rates), -1)
        info["disturbance_estimate"] = self.disturbance_estimate.clone()
        info["current_estimate_ned"] = self.current_estimate.clone()
        return self.observe(), reward, done, info
