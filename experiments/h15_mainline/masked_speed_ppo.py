"""Five-intent masked PPO policy and deployable H15 decision features."""

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from experiments.h15_mainline.candidate_filter import SCALE


GRID = (13., 14., 15., 16., 17.)
OBSERVATION_SIZE = 60


def _response_residual(previous_observation, dynamics):
    if previous_observation is None:
        return np.zeros(4, dtype=float)
    import json

    row = previous_observation
    before = np.asarray(json.loads(row["state_before_json"]), dtype=float)
    after = np.asarray(json.loads(row["state_after_json"]), dtype=float)
    actuator = np.asarray(json.loads(row["actuator_before_json"]), dtype=float)
    queue = np.asarray(json.loads(row["queue_before_json"]), dtype=float)
    command = np.asarray(json.loads(row["command_json"]), dtype=float)
    actual_after = np.asarray([
        row["thrust_actual_n"], math.radians(row["elevator_actual_deg"]),
        math.radians(row["rudder_actual_deg"])], dtype=float)
    if (before.shape != (10,) or after.shape != (10,) or actuator.shape != (3,) or
            queue.shape != (5, 3) or command.shape != (3,)):
        raise ValueError("invalid prior observed response")
    augmented = np.concatenate((before, actuator/SCALE, (queue[:3]/SCALE).ravel()))
    predicted = dynamics.evaluate(augmented[None], (command/SCALE)[None])[0][0]
    residual = np.asarray([(after[8]-predicted[8])*10.,
                           (after[9]-predicted[9])*10.,
                           actual_after[1]/SCALE[1]-predicted[11],
                           actual_after[2]/SCALE[2]-predicted[12]])
    if not np.isfinite(residual).all():
        raise ValueError("nonfinite recent response residual")
    return residual


def decision_observation(state, sample, actuator, queue, path, progress,
                         estimate, candidates, held_intent,
                         previous_observation=None, dynamics=None):
    """Use only measured state, known geometry, fit output and candidate results."""
    state = np.asarray(state, dtype=float)
    actuator = np.asarray(actuator, dtype=float)
    queue = np.asarray(queue, dtype=float)
    if state.shape != (10,) or actuator.shape != (3,) or queue.shape != (5, 3):
        raise ValueError("invalid deployable decision state")
    if tuple(item["candidate_mps"] for item in candidates) != GRID:
        raise ValueError("candidate grid differs from policy grid")
    if previous_observation is not None and dynamics is None:
        raise ValueError("previous response requires the current prediction model")
    ahead = min(float(path.distances[-1]), float(progress) + 20.)
    tangent = path.tangent_at(float(progress))
    tangent_ahead = path.tangent_at(ahead)
    yaw_delta = math.atan2(tangent[0]*tangent_ahead[1]-tangent[1]*tangent_ahead[0],
                           tangent[0]*tangent_ahead[0]+tangent[1]*tangent_ahead[1])
    depth_slope = (path.position_at(ahead)[2]-path.position_at(float(progress))[2]) / 20.
    costs = [max(-10., min(10., float(item["cost"]))) if item.get("cost") is not None
             else 10. for item in candidates]
    risks = [min(2., max(0., max((float(s["risk"]) for s in item["scenarios"]),
                                   default=2.))) for item in candidates]
    limits = (8., 5., 40., 2.)
    margin_fields = ("lateral_m", "depth_m", "pitch_deg",
                     "tangent_water_speed_error_mps")
    margins = [max(-2., min(2., min(
        (float(s["min_margins"][field])/limit for s in item["scenarios"]
         for field, limit in zip(margin_fields, limits)), default=-2.)))
        for item in candidates]
    qp_valid = [float(item["qp_valid"]) for item in candidates]
    mask = [bool(item["physical_accepted"]) for item in candidates]
    residual = _response_residual(previous_observation, dynamics)
    values = [sample["lateral_error_m"]/8., sample["depth_error_m"]/5.,
              sample["pitch_deg"]/40., sample["speed_error_mps"]/2.,
              state[8]*10., state[9]*10.,
              *(actuator/SCALE), *((queue/SCALE).ravel()),
              yaw_delta, depth_slope,
              *residual,
              estimate.fin_gain-.85, estimate.fin_tau_scale/2.,
              estimate.gain_radius, estimate.tau_radius/2.,
              (held_intent-15.)/2., *costs, *risks, *margins, *qp_valid,
              *map(float, mask)]
    observation = np.asarray(values, dtype=np.float32)
    if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
        raise ValueError("nonfinite or malformed policy observation")
    return observation, np.asarray(mask, dtype=bool)


class MaskedSpeedActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(OBSERVATION_SIZE, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 5))
        self.critic = nn.Sequential(nn.Linear(OBSERVATION_SIZE, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, observation, mask):
        if observation.shape[-1] != OBSERVATION_SIZE or mask.shape != observation.shape[:-1] + (5,):
            raise ValueError("policy observation or action mask shape differs")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("masked policy cannot select from an empty candidate set")
        return Categorical(logits=self.actor(observation).masked_fill(~mask, -1e9))

    def evaluate(self, observation, mask, action):
        distribution = self.distribution(observation, mask)
        return (distribution.log_prob(action), distribution.entropy(),
                self.critic(observation).squeeze(-1))

    @torch.no_grad()
    def choose(self, observation, mask, deterministic=False):
        observation = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        mask = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        distribution = self.distribution(observation, mask)
        action = (distribution.logits.argmax(-1) if deterministic else
                  distribution.sample())
        return (int(action.item()), float(distribution.log_prob(action).item()),
                float(self.critic(observation).squeeze().item()))
