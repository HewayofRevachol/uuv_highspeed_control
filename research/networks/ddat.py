"""Checkpoint-compatible DDAT actor and critic architecture."""

import torch
from torch import nn
from torch.distributions import Normal

from simulator.ddat_env import DDATPathEnv


def network(input_dim, output_dim):
    layers = []
    for width in (256, 256):
        layers += [nn.Linear(input_dim, width), nn.ReLU()]
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, observation_dim):
        super().__init__()
        if observation_dim not in (DDATPathEnv.preview_size + 35,
                                   DDATPathEnv.observation_size):
            raise ValueError("DDAT observation dimensions do not match")
        self.observation_dim = observation_dim
        self.preview_size = DDATPathEnv.preview_size
        self.feedforward_encoder = nn.Sequential(
            nn.Linear(self.preview_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU())
        feature_dim = 64 + observation_dim-self.preview_size
        self.actor = network(feature_dim, 6)
        self.critic = network(feature_dim, 1)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():
            self.actor[-1].bias[3:] = -1.0

    def features(self, observation):
        # Existing checkpoints were trained before the five MPC diagnostics
        # were appended to the observation.
        observation = observation[..., :self.observation_dim]
        preview = observation[..., :self.preview_size]
        auxiliary = observation[..., self.preview_size:]
        return torch.cat((self.feedforward_encoder(preview), auxiliary), -1)

    def distribution(self, observation):
        mean, log_std = self.actor(self.features(observation)).chunk(2, dim=-1)
        return Normal(mean, log_std.clamp(-5., 2.).exp())

    def deterministic(self, observation):
        return torch.tanh(self.actor(self.features(observation))[..., :3])

    def value(self, observation):
        return self.critic(self.features(observation)).squeeze(-1)

    def sample(self, observation):
        distribution = self.distribution(observation)
        raw = distribution.sample()
        action = torch.tanh(raw)
        logp = (distribution.log_prob(raw) -
                torch.log(1-action.square()+1e-6)).sum(-1)
        return action, logp, self.value(observation)

    def log_prob(self, observation, action):
        clipped = action.clamp(-1+1e-6, 1-1e-6)
        raw = torch.atanh(clipped)
        return (self.distribution(observation).log_prob(raw) -
                torch.log(1-clipped.square()+1e-6)).sum(-1)


