"""Small categorical PPO update using the repository's GAE helper."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from training.ppo import advantages as gae_advantages


class BetaActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(19, 64), nn.Tanh(),
                                  nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 4)
        self.critic = nn.Linear(64, 1)

    def distribution(self, observations):
        return Categorical(logits=self.actor(self.body(observations)))

    def value(self, observations):
        return self.critic(self.body(observations)).squeeze(-1)


def update(agent, optimizer, batch, epochs=4, minibatch=64, clip=.2,
           value_weight=.5, entropy_weight=.01, grad_norm=1.):
    """One on-policy update; batch=(obs, action, old_logp, returns, advantage)."""
    obs, actions, old_logp, returns, adv = batch
    count = len(obs)
    if obs.shape != (count, 19) or any(x.shape != (count,) for x in
                                         (actions, old_logp, returns, adv)):
        raise ValueError("categorical PPO batch has wrong shape")
    if count == 0 or not all(torch.isfinite(x).all() for x in
                             (obs, old_logp, returns, adv)):
        raise ValueError("categorical PPO batch is empty or nonfinite")
    if actions.dtype != torch.long or (actions < 0).any() or (actions > 3).any():
        raise ValueError("categorical actions must be int64 in [0, 3]")
    normalized_adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    metrics = []
    for _ in range(epochs):
        for indices in torch.randperm(count, device=obs.device).split(minibatch):
            sub_obs = obs[indices]
            sub_actions = actions[indices]
            distribution = agent.distribution(sub_obs)
            new_logp = distribution.log_prob(sub_actions)
            ratio = (new_logp-old_logp[indices]).exp()
            policy_loss = -torch.minimum(
                ratio*normalized_adv[indices],
                ratio.clamp(1.-clip, 1.+clip)*normalized_adv[indices]).mean()
            values = agent.value(sub_obs)
            value_loss = (values-returns[indices]).square().mean()
            entropy = distribution.entropy().mean()  # categorical shape: [batch]
            loss = policy_loss + value_weight*value_loss-entropy_weight*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite categorical PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(agent.parameters(), grad_norm,
                                                error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in agent.parameters()):
                raise FloatingPointError("nonfinite categorical PPO parameter")
            metrics.append((
                float(policy_loss.detach()), float(value_loss.detach()),
                float(entropy.detach()),
                float((old_logp[indices]-new_logp).mean().detach()),
                float(((ratio-1.).abs() > clip).float().mean().detach()),
                float(gradient.detach())))
    return {
        "policy_loss": sum(x[0] for x in metrics)/len(metrics),
        "value_loss": sum(x[1] for x in metrics)/len(metrics),
        "entropy": sum(x[2] for x in metrics)/len(metrics),
        "approx_kl": sum(x[3] for x in metrics)/len(metrics),
        "clip_fraction": sum(x[4] for x in metrics)/len(metrics),
        "gradient_norm_mean": sum(x[5] for x in metrics)/len(metrics),
        "gradient_norm_max": max(x[5] for x in metrics),
        "minibatch_updates": len(metrics),
    }
