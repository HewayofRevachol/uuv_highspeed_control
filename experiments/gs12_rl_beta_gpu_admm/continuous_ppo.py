"""Continuous-beta actor/critic and clipped PPO for the CUDA lane batch."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class ContinuousBetaActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(19, 64), nn.Tanh(),
                                  nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 1)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.tensor([-1.]))

    def distribution(self, observation):
        mean = self.actor(self.body(observation)).squeeze(-1)
        return Normal(mean, self.log_std.clamp(-5., 1.).exp().expand_as(mean))

    def value(self, observation):
        return self.critic(self.body(observation)).squeeze(-1)

    @staticmethod
    def target(raw):
        return .75*torch.sigmoid(raw)

    def sample(self, observation):
        distribution = self.distribution(observation)
        raw = distribution.sample()
        return raw, self.target(raw), distribution.log_prob(raw), self.value(observation)

    def deterministic(self, observation):
        return self.target(self.distribution(observation).mean)


def gae_batch(observations, raw, logp, values, rewards, done, valid,
              gamma=.98, lam=.95):
    """Compute GAE independently for each lane, then select live transitions."""
    values = torch.stack(values)
    rewards = torch.stack(rewards)
    done = torch.stack(done)
    valid = torch.stack(valid)
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(len(rewards)-1, -1, -1):
        next_value = values[t+1] if t+1 < len(rewards) else torch.zeros_like(values[t])
        continuing = (~done[t]).to(rewards.dtype)
        delta = rewards[t]+gamma*next_value*continuing-values[t]
        carry = (delta+gamma*lam*continuing*carry)*valid[t]
        advantage[t] = carry
    returns = advantage+values
    mask = valid.flatten()
    return (torch.stack(observations).reshape(-1, 19)[mask],
            torch.stack(raw).flatten()[mask],
            torch.stack(logp).flatten()[mask],
            returns.flatten()[mask], advantage.flatten()[mask])


def update(agent, optimizer, batch, epochs=4, minibatch=64):
    observations, raw, old_logp, returns, advantages = batch
    n = len(observations)
    if n == 0 or observations.shape != (n, 19):
        raise ValueError("empty or invalid PPO batch")
    if not all(bool(torch.isfinite(item).all()) for item in batch):
        raise FloatingPointError("nonfinite PPO batch")
    advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(n, device=observations.device).split(minibatch):
            distribution = agent.distribution(observations[indices])
            ratio = (distribution.log_prob(raw[indices])-old_logp[indices]).exp()
            policy_loss = -torch.minimum(ratio*advantages[indices],
                ratio.clamp(.8, 1.2)*advantages[indices]).mean()
            value_loss = (agent.value(observations[indices])-returns[indices]).square().mean()
            loss = policy_loss+.5*value_loss-.01*distribution.entropy().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    return {"loss": sum(losses)/len(losses), "minibatch_steps": len(losses)}
