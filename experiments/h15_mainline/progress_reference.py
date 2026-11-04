"""Single speed-intent reference for H15 engineering smoke runs.

This changes only the geometric MPC reference. It is not a feasibility
filter, safety governor, or RL policy.
"""
import math

import torch

from aav_mpc.controller import EndpointExtendedCondensedEnv


class ProgressReferenceEnv(EndpointExtendedCondensedEnv):
    def reset(self, path_ids, mask=None):
        if self.batch_size != 1:
            raise ValueError("H15 progress reference currently supports batch_size=1")
        self.intent_mps = 15.
        self._reference_step = 0
        self._reference_value = 15.
        return super().reset(path_ids, mask)

    def set_intent(self, nu_ref_mps):
        value = float(nu_ref_mps)
        if not math.isfinite(value) or not 13. <= value <= 17.:
            raise ValueError("nu_ref must be in [13, 17] m/s")
        self.intent_mps = value

    def reference_rate(self):
        intent = getattr(self, "intent_mps", 15.)
        steps = getattr(self, "steps", None)
        count = 0 if steps is None else int(steps[0])
        prior_step = getattr(self, "_reference_step", 0)
        if count < prior_step:
            prior_step = 0
            self._reference_value = 15.
        value = getattr(self, "_reference_value", 15.)
        for _ in range(count - prior_step):
            value += max(-.05, min(.05, intent - value))
        self._reference_step = count
        self._reference_value = value
        return value

    def _reference(self, time_offset=0.):
        if time_offset != 0.:
            raise ValueError("time-offset references are not used in H15 mainline")
        current = self.reference_rate()
        desired = getattr(self, "intent_mps", 15.)
        horizon = self.mpc.horizon
        dt = self.mpc.dt
        indices = torch.arange(1, horizon + 1, device=self.device, dtype=self.dtype)
        nodes = torch.as_tensor(current, device=self.device, dtype=self.dtype) + \
            torch.as_tensor(desired - current, device=self.device, dtype=self.dtype).clamp(
                -.05 * indices, .05 * indices)
        nodes = nodes.clamp(13., 17.)
        distances = self.progress[:, None] + dt * nodes.cumsum(0)[None]
        position, tangent, pitch, yaw, _ = self.paths.sample(self.path_ids, distances)
        overflow = (distances - self.paths.lengths[self.path_ids, None]).clamp_min(0.)
        _, endpoint_tangent, _, _, _ = self.paths.sample(
            self.path_ids, self.paths.lengths[self.path_ids, None])
        position = position + overflow[..., None] * endpoint_tangent[:, :1]
        velocity = nodes[None, :, None] * tangent
        offsets = dt * indices
        reference = torch.cat((position, pitch[..., None], yaw[..., None], velocity), -1)
        reference[..., :3] -= offsets[None, :, None] * self.current_estimate[:, None]
        reference[..., 5:8] -= self.current_estimate[:, None]
        return reference
