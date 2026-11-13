"""78 -> 64 -> 64 actor/critic. PPO ratios use the stored Gaussian latent."""
import math
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(78, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor, self.critic = nn.Linear(64, 1), nn.Linear(64, 1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        self.log_std = nn.Parameter(torch.tensor(math.log(.15)))

    def distribution(self, observation):
        mean = self.actor(self.body(observation)).squeeze(-1)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, observation):
        return self.critic(self.body(observation)).squeeze(-1)

    @staticmethod
    def target(latent):
        return .5+.5*latent.tanh()

    def sample(self, observation):
        distribution = self.distribution(observation)
        latent = distribution.sample()
        return latent, self.target(latent), distribution.log_prob(latent), self.value(observation)

    def deterministic(self, observation):
        return self.target(self.distribution(observation).mean)


def advantages(values, rewards, done, valid, lam=.95):
    """Full finite episodes, gamma=1, independently masked lanes."""
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(len(rewards)-1, -1, -1):
        following = values[t+1] if t+1 < len(values) else torch.zeros_like(carry)
        continuation = (~done[t]).float()
        carry = (rewards[t]+following*continuation-values[t]+lam*continuation*carry)*valid[t]
        advantage[t] = carry
    return advantage, advantage+values


def update(agent, optimizer, batch):
    observation, latent, logp, returns, advantage = batch
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    losses = []
    for _ in range(4):
        for index in torch.randperm(len(observation), device=observation.device).split(64):
            distribution = agent.distribution(observation[index])
            ratio = (distribution.log_prob(latent[index])-logp[index]).exp()
            objective = torch.minimum(ratio*advantage[index], ratio.clamp(.8, 1.2)*advantage[index])
            loss = -objective.mean()+.5*(agent.value(observation[index])-returns[index]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
    return sum(losses)/len(losses)
