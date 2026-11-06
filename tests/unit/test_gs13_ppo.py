"""Regression tests for the optimizer's actual safety and rollout semantics."""
import math

import pytest
import torch

from experiments.gs13_srppo.policy import SRPPOActorCritic
from experiments.gs13_srppo.ppo import (LagrangeState, constrained_advantage,
                                         gae, srppo_update)


def rollout(policy, count=8):
    torch.manual_seed(7)
    observation = torch.randn(count, policy.observation_dim) * .1
    next_observation = observation + .15
    with torch.no_grad():
        action = policy.act(observation)
        next_value = policy.values(next_observation)
    return {"observations": observation, "next_observations": next_observation,
            "raw_z": action["raw_z"], "old_logp": action["log_prob"],
            "rewards": -torch.arange(count).float() / count,
            "safety_cost": torch.rand(count, 4) * .1,
            "value_track": action["value_track"], "value_safe": action["value_safe"],
            "next_value_track": next_value["value_track"],
            "next_value_safe": next_value["value_safe"],
            "terminated": torch.zeros(count, dtype=torch.bool),
            "truncated": torch.tensor([False] * (count - 1) + [True])}


def test_positive_cost_advantage_reduces_objective_by_its_own_multiplier():
    reward = torch.zeros(2)
    costs = torch.tensor([[0., 1., 0., 0.], [0., 0., 1., 0.]])
    objective = constrained_advantage(reward, costs, torch.tensor([0., 2., 5., 0.]))
    torch.testing.assert_close(objective, torch.tensor([-2., -5.]))


def test_cost_penalty_decreases_probability_of_high_cost_action():
    torch.manual_seed(1)
    policy = SRPPOActorCritic(hidden=16)
    obs = torch.zeros(2, 200)
    actions = torch.tensor([[-.5, 0., 0., 0.], [.5, 0., 0., 0.]])
    with torch.no_grad():
        old = policy.log_prob(obs, actions)
    batch = {"observations": obs, "raw_z": actions, "old_logp": old,
             "rewards": torch.zeros(2), "safety_cost": torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.]]),
             "value_track": torch.zeros(2), "value_safe": torch.zeros(2, 4),
             "next_value_track": torch.zeros(2), "next_value_safe": torch.zeros(2, 4),
             "terminated": torch.ones(2, dtype=torch.bool),
             "truncated": torch.zeros(2, dtype=torch.bool)}
    state = LagrangeState.create(4, [0.] * 4, learning_rate=0.)
    state.lambdas[0] = 2.
    srppo_update(policy, torch.optim.SGD(policy.parameters(), lr=.01), batch, state,
                 epochs=1, minibatch=2, value_weight=0., entropy_weight=0., gamma=1.)
    with torch.no_grad():
        after = policy.log_prob(obs, actions)
    assert after[1] < old[1]  # high-cost action is discouraged
    assert after[0] > old[0]


def test_vector_critics_checkpoint_architecture_and_finite_update():
    policy = SRPPOActorCritic(hidden=32, residual_limit=.5)
    assert policy.variant == "srppo"
    restored = SRPPOActorCritic.from_checkpoint_config(policy.checkpoint_config())
    restored.load_state_dict(policy.state_dict())
    batch = rollout(policy)
    assert batch["value_safe"].shape == (8, 4)
    with torch.no_grad():
        torch.testing.assert_close(policy.act(batch["observations"], deterministic=True)["alpha_raw"],
                                   restored.act(batch["observations"], deterministic=True)["alpha_raw"])
    metrics = srppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-4), batch,
                            LagrangeState.create(4, [.05] * 4), epochs=1, minibatch=4)
    assert all(math.isfinite(value) for value in metrics.values())
    assert all(f"lambda_{i}" in metrics for i in range(4))


def test_vector_gae_bootstraps_truncation_without_crossing_reset():
    rewards = torch.zeros(3, 4)
    values = torch.zeros(3, 4)
    next_values = torch.tensor([[1., 2., 3., 4.], [9., 9., 9., 9.], [5., 6., 7., 8.]])
    done = torch.tensor([True, True, True])
    terminated = torch.tensor([False, True, False])
    advantage, returns = gae(rewards, values, next_values, done, terminated, gamma=.5, lam=1.)
    torch.testing.assert_close(advantage[0], next_values[0] * .5)
    torch.testing.assert_close(advantage[1], torch.zeros(4))
    torch.testing.assert_close(advantage[2], next_values[2] * .5)
    torch.testing.assert_close(returns, advantage)


def test_old_rollout_data_detached_and_next_values_mandatory():
    policy = SRPPOActorCritic(hidden=16)
    batch = rollout(policy, 4)
    batch["old_logp"].requires_grad_()
    batch["value_track"].requires_grad_()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    srppo_update(policy, optimizer, batch, epochs=1, minibatch=4)
    assert batch["old_logp"].grad is None and batch["value_track"].grad is None
    for field in ("next_value_track", "next_value_safe", "next_observations"):
        del batch[field]
    with pytest.raises(KeyError, match="next_value"):
        srppo_update(policy, optimizer, batch, epochs=1)
    batch["next_observations"] = batch["observations"] + .1
    metrics = srppo_update(policy, optimizer, batch, epochs=1, minibatch=4)
    assert math.isfinite(metrics["value_loss"])


def test_behavior_cloning_nonidentity_targets_have_actor_gradient_and_reduce_loss():
    torch.manual_seed(3)
    policy = SRPPOActorCritic(hidden=16, residual_limit=1.)
    obs = torch.ones(4, 200) * .1
    targets = torch.tensor([[.4, .6, .7, .3]]).expand(4, -1)
    loss = policy.behavior_cloning_loss(obs, None, targets)
    loss.backward()
    assert policy.actor_mean.bias.grad.abs().sum() > 0
    initial = loss.item()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
    for _ in range(8):
        optimizer.zero_grad()
        loss = policy.behavior_cloning_loss(obs, None, targets)
        loss.backward()
        optimizer.step()
    assert policy.behavior_cloning_loss(obs, None, targets).item() < initial
