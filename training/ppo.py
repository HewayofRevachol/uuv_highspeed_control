"""PPO objective and time-limit-aware advantage estimation."""

import torch
from torch import nn


def advantages(rewards, values, next_values, done, terminated,
               gamma=.99, lam=.95):
    """Bootstrap time limits; stop GAE recursion across every reset."""
    result = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma*next_values[t]*(~terminated[t]) - values[t]
        gae = delta + gamma*lam*(~done[t])*gae
        result[t] = gae
    return result, result+values


def ppo_update(agent, optimizer, batch, epochs=4, minibatch=2048,
               clip=.2, value_weight=.5, entropy_weight=.001):
    observations, actions, old_logp, returns, adv = batch
    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    count = len(observations)
    for _ in range(epochs):
        for indices in torch.randperm(count, device=observations.device).split(minibatch):
            obs, act = observations[indices], actions[indices]
            logp = agent.log_prob(obs, act)
            ratio = (logp-old_logp[indices]).exp()
            policy_loss = -torch.minimum(ratio*adv[indices],
                                         ratio.clamp(1-clip, 1+clip)*adv[indices]).mean()
            value = agent.value(obs)
            value_loss = (value-returns[indices]).square().mean()
            entropy = agent.distribution(obs).entropy().sum(-1).mean()
            loss = policy_loss + value_weight*value_loss-entropy_weight*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), .5)
            optimizer.step()
