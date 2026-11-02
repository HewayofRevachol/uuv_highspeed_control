"""DDAT experiment environment: preview layout and course reward."""

import numpy as np
import torch

from simulator.ddat_env import DDATPathEnv

MULTISCALE_PREVIEW_OFFSETS_M = (0., 4., 8., 12., 20., 28., 40., 56.,
                                72., 96., 120., 160.)


class ExperimentDDATPathEnv(DDATPathEnv):
    """Keep the existing actor shape while changing preview and turn reward."""

    def __init__(self, *args, preview_mode="dense", yaw_reward="heading", **kwargs):
        if preview_mode not in ("dense", "multiscale"):
            raise ValueError("preview_mode must be dense or multiscale")
        if yaw_reward not in ("heading", "course"):
            raise ValueError("yaw_reward must be heading or course")
        self.preview_mode = preview_mode
        self.yaw_reward = yaw_reward
        self._course_reward_active = False
        super().__init__(*args, **kwargs)

    def _errors(self):
        errors = super()._errors()
        if not self._course_reward_active:
            return errors
        # Reward the direction of travel over ground, allowing the sideslip
        # needed to turn or compensate current. Observations below retain the
        # original heading error, so true current is not exposed to the actor.
        rotation = errors[-1]
        ground_velocity = torch.bmm(rotation, self.state[:, 5:8, None]).squeeze(-1)
        ground_velocity = ground_velocity + self.current_ned
        course = torch.atan2(ground_velocity[:, 1], ground_velocity[:, 0])
        _, _, _, path_yaw, _ = self.paths.sample(
            self.path_ids, self.progress[:, None])
        yaw = torch.atan2(torch.sin(path_yaw[:, 0]-course),
                          torch.cos(path_yaw[:, 0]-course))
        return errors[:3] + (yaw,) + errors[4:]

    @torch.no_grad()
    def observe(self):
        # Parent step calls observe before returning. Keep course reward active
        # only for the five integration substeps, never for policy features.
        reward_active = self._course_reward_active
        self._course_reward_active = False
        try:
            observation = super().observe()
            if self.preview_mode == "dense":
                return observation
            offsets = self.model.tensor(MULTISCALE_PREVIEW_OFFSETS_M)
            requested = self.progress[:, None] + offsets[None]
            valid = requested <= self.paths.lengths[self.path_ids, None]
            points, _, path_pitch, path_yaw, _ = self.paths.sample(
                self.path_ids, requested)
            rotation = self._errors()[-1]
            body_preview = torch.bmm(points-self.state[:, None, :3], rotation)
            relative_yaw = torch.atan2(
                torch.sin(path_yaw-self.state[:, 4:5]),
                torch.cos(path_yaw-self.state[:, 4:5]))
            preview = torch.stack((
                body_preview[..., 0]/50, body_preview[..., 1]/50,
                body_preview[..., 2]/50, relative_yaw/np.pi,
                (path_pitch-self.state[:, 3:4])/.5,
                valid.to(self.dtype)), -1).flatten(1).float()
            return torch.cat((preview, observation[:, self.preview_size:]), -1)
        finally:
            self._course_reward_active = reward_active

    @torch.no_grad()
    def step(self, action):
        self._course_reward_active = self.yaw_reward == "course"
        try:
            return super().step(action)
        finally:
            self._course_reward_active = False


