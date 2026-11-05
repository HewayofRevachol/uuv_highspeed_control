"""Continuous two-branch Gaussian gate; PPO density is over raw z."""
import torch
from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, observation_dim=37, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 2))
        self.critic = nn.Sequential(nn.Linear(observation_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(),
                                    nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.zeros(2))

    def distribution(self, observation):
        mean = self.actor(observation)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def act(self, observation, alpha_previous=None, slew=.25, deterministic=False):
        distribution = self.distribution(observation)
        raw_z = distribution.mean if deterministic else distribution.rsample()
        log_prob = distribution.log_prob(raw_z).sum(-1)
        target = torch.sigmoid(raw_z)
        if alpha_previous is None:
            applied = target.clamp(0., 1.)
        else:
            applied = alpha_previous + (target-alpha_previous).clamp(-slew, slew)
        return {"raw_z": raw_z, "log_prob": log_prob,
                "value": self.critic(observation).squeeze(-1),
                "alpha_target": target, "alpha": applied}

    def evaluate_raw(self, observation, raw_z):
        distribution = self.distribution(observation)
        return (distribution.log_prob(raw_z).sum(-1),
                distribution.entropy().sum(-1),
                self.critic(observation).squeeze(-1))
