"""Gaussian continuous-beta policy and clipped PPO update."""
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

    def distribution(self, obs):
        mean = self.actor(self.body(obs)).squeeze(-1)
        return Normal(mean, self.log_std.clamp(-5., 1.).exp().expand_as(mean))

    def value(self, obs):
        return self.critic(self.body(obs)).squeeze(-1)

    @staticmethod
    def target(raw):
        return .75 * torch.sigmoid(raw)

    def sample(self, obs):
        dist = self.distribution(obs)
        raw = dist.sample()
        return raw, self.target(raw), dist.log_prob(raw), self.value(obs)

    def deterministic(self, obs):
        return self.target(self.distribution(obs).mean)


def update(agent, optimizer, batch, epochs=4, minibatch=64):
    obs, raw, old_logp, returns, advantages = batch
    n = len(obs)
    if obs.shape != (n, 19) or n == 0 or any(
            x.shape != (n,) for x in (raw, old_logp, returns, advantages)):
        raise ValueError("invalid continuous PPO batch shape")
    if not all(torch.isfinite(x).all() for x in batch):
        raise ValueError("nonfinite continuous PPO batch")
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    losses = []
    for _ in range(epochs):
        for ix in torch.randperm(n, device=obs.device).split(minibatch):
            dist = agent.distribution(obs[ix])
            logp = dist.log_prob(raw[ix])
            ratio = (logp - old_logp[ix]).exp()
            policy_loss = -torch.minimum(ratio * advantages[ix],
                ratio.clamp(.8, 1.2) * advantages[ix]).mean()
            value_loss = (agent.value(obs[ix]) - returns[ix]).square().mean()
            loss = policy_loss + .5 * value_loss - .01 * dist.entropy().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite continuous PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    if not all(torch.isfinite(p).all() for p in agent.parameters()):
        raise FloatingPointError("nonfinite continuous PPO parameters")
    return {"loss": sum(losses) / len(losses), "minibatch_steps": len(losses)}
