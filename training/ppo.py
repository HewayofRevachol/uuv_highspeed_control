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
               clip=.2, value_weight=.5, entropy_weight=.001,
               grad_norm=.5):
    observations, actions, old_logp, returns, adv = batch
    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    count = len(observations)
    metrics = []
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
            gradient = nn.utils.clip_grad_norm_(agent.parameters(), grad_norm)
            optimizer.step()
            metrics.append((float(policy_loss.detach()), float(value_loss.detach()),
                            float(entropy.detach()), float((old_logp[indices]-logp).mean().detach()),
                            float(((ratio-1).abs() > clip).float().mean().detach()),
                            float(gradient.detach())))
    with torch.no_grad():
        value = agent.value(observations)
        variance = returns.var(unbiased=False)
        explained = (1-(returns-value).var(unbiased=False)/variance
                     if variance > 1e-12 else returns.new_tensor(0.))
    return {"policy_loss": sum(x[0] for x in metrics)/len(metrics),
            "value_loss": sum(x[1] for x in metrics)/len(metrics),
            "entropy": sum(x[2] for x in metrics)/len(metrics),
            "approx_kl": sum(x[3] for x in metrics)/len(metrics),
            "clip_fraction": sum(x[4] for x in metrics)/len(metrics),
            "gradient_norm_mean": sum(x[5] for x in metrics)/len(metrics),
            "gradient_norm_max": max(x[5] for x in metrics),
            "explained_variance": float(explained)}
