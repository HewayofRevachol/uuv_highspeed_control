"""RTI interface with an observed actuator and command-history state."""
import torch

from high_speed_experiments.reference_schedule_probe import ScheduledReferenceEnv
from simulator.rti_mpc import BatchedRTIMPC
from aav_mpc.condensed_mpc import CondensedActuatorMPC


class ActuatorAwareEnv(ScheduledReferenceEnv):
    def __init__(self, *args, **kwargs):
        if kwargs.get("backend", "rti_cpu") != "rti_cpu":
            raise ValueError("ActuatorAwareEnv currently supports rti_cpu only")
        kwargs["fsp_enabled"] = False
        super().__init__(*args, **kwargs)
        self.mpc = BatchedRTIMPC(
            self.model, self.batch_size, horizon=self.mpc.horizon,
            dt=self.mpc.dt, backend="rti_cpu", actuator_aware=True,
            attitude_pitch_weight=self.mpc.attitude_pitch_weight,
            attitude_yaw_weight=self.mpc.attitude_yaw_weight)

    def observed_augmented_state(self):
        return torch.cat((self.state,
            self.actuator / self.model.control_scale,
            self.command_history[:, :3].reshape(self.batch_size, -1) /
                self.model.control_scale.repeat(3)), -1)

    def _solve_mpc(self):
        return self.mpc.solve(self.observed_augmented_state(), self._reference())


class EndpointExtensionMixin:
    """Continue the final path tangent for MPC reference beyond the endpoint.

    The completion test remains the original endpoint and original task speed.
    This only removes the artificial zero-progress target at the end of the
    finite CSV reference; it uses no future information unavailable to peers.
    """

    def _reference(self, time_offset=0.):
        reference = super()._reference(time_offset)
        offsets = torch.arange(1, self.mpc.horizon + 1, device=self.device,
                               dtype=self.dtype) * self.mpc.dt
        _, _, _, _, speed = self.paths.sample(self.path_ids, self.progress[:, None])
        distances = self.progress[:, None] + speed.clamp_min(.1) * (
            self.preview_scale * offsets[None] + time_offset)
        overflow = (distances - self.paths.lengths[self.path_ids, None]).clamp_min(0)
        _, tangent, _, _, _ = self.paths.sample(
            self.path_ids, self.paths.lengths[self.path_ids, None])
        reference[..., :3] += overflow[..., None] * tangent[:, :1, :]
        return reference


class EndpointExtendedEnv(EndpointExtensionMixin, ScheduledReferenceEnv):
    pass


class EndpointExtendedAwareEnv(EndpointExtensionMixin, ActuatorAwareEnv):
    pass


class CondensedActuatorEnv(ActuatorAwareEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mpc = CondensedActuatorMPC(self.model, self.batch_size,
            horizon=self.mpc.horizon, dt=self.mpc.dt,
            attitude_pitch_weight=self.mpc.attitude_pitch_weight,
            attitude_yaw_weight=self.mpc.attitude_yaw_weight)


class EndpointExtendedCondensedEnv(EndpointExtensionMixin, CondensedActuatorEnv):
    pass
