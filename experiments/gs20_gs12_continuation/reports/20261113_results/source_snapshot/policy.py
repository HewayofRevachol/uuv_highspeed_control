"""Small Gaussian PPO policy; sigmoid maps actions to the full beta interval."""
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(19, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor, self.critic = nn.Linear(64, 1), nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.tensor([-1.]))

    def distribution(self, obs):
        mean = self.actor(self.body(obs)).squeeze(-1)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs):
        return self.critic(self.body(obs)).squeeze(-1)

    def target(self, obs):
        return self.distribution(obs).mean.sigmoid()


def update(agent, optimizer, rows, epochs=4):
    obs, raw, old_logp, returns, advantage = zip(*rows)
    obs = torch.as_tensor(__import__('numpy').asarray(obs), dtype=torch.float32)
    raw, old_logp, returns, advantage = [torch.tensor(x, dtype=torch.float32)
                                       for x in (raw, old_logp, returns, advantage)]
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    losses = []
    for _ in range(epochs):
        for ix in torch.randperm(len(rows)).split(64):
            distribution = agent.distribution(obs[ix])
            ratio = (distribution.log_prob(raw[ix])-old_logp[ix]).exp()
            actor = -torch.minimum(ratio*advantage[ix], ratio.clamp(.8, 1.2)*advantage[ix]).mean()
            critic = (agent.value(obs[ix])-returns[ix]).square().mean()
            loss = actor+.5*critic-.01*distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
    return sum(losses)/len(losses)
