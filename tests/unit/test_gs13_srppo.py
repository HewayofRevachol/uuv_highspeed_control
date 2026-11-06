import numpy as np
import pytest
import torch

from experiments.gs13_srppo.costs import aggregate_cost, compute_cost
from experiments.gs13_srppo.features import CausalObservation
from experiments.gs13_srppo.policy import SRPPOActorCritic
from experiments.gs13_srppo.ppo import LagrangeState, gae
from experiments.gs13_srppo.safety import SafetyConfig, project_gate
from experiments.gs13_srppo.sampler import StratifiedSampler
from experiments.gs13_srppo.teacher import build_teacher_label


def test_projection_bounds_slew_and_margin():
    result = project_gate(np.zeros(4), np.ones(4), {"teacher_gate": np.ones(4), "risk": .9})
    assert np.all((result.alpha >= 0.) & (result.alpha <= 1.))
    assert np.all(np.abs(result.alpha) <= .25 + 1e-9)
    assert result.overridden and "risk" in result.reason


def test_projection_qp_and_nan_fail_closed():
    assert project_gate(np.zeros(4), np.ones(4), {"qp_success": False}).emergency
    result = project_gate(np.zeros(4), np.array([np.nan] * 4))
    assert result.emergency and result.reason == "nonfinite_input"


def test_teacher_residual_is_centered_and_logprob_uses_raw_z():
    torch.manual_seed(1)
    policy = SRPPOActorCritic()
    obs = torch.zeros(2, 200)
    teacher = torch.tensor([1., 1., 1., 0.])
    sample = policy.act(obs, teacher, deterministic=True)
    assert sample["raw_z"].shape == (2, 4)
    assert torch.allclose(sample["alpha_raw"], teacher.expand(2, -1))
    assert torch.isfinite(policy.log_prob(obs, sample["raw_z"])).all()


def test_barrier_and_failure_truncation_semantics():
    safe = compute_cost(None, None, np.zeros(4), np.zeros(4), np.zeros(3),
                        {"errors": np.zeros(4), "actual": np.zeros(3)})
    unsafe = compute_cost(None, None, np.zeros(4), np.zeros(4), np.zeros(3),
                          {"errors": np.array([9., 0., 0., 0.]), "actual": np.zeros(3),
                           "failed": True, "terminated": True, "remaining_s": 2.})
    trunc = compute_cost(None, None, np.zeros(4), np.zeros(4), np.zeros(3),
                         {"errors": np.zeros(4), "truncated": True, "failed": True,
                          "terminated": False, "remaining_s": 2.})
    assert unsafe.boundary > safe.boundary and unsafe.terminal == 220.
    assert trunc.terminal == 0. and trunc.failed == 1.
    assert aggregate_cost([safe, unsafe])["terminal"] == 220.


def test_gae_bootstraps_truncation_and_lagrange_clamps():
    rewards = torch.tensor([0., 0.])
    values = torch.zeros(2); next_values = torch.tensor([0., 2.])
    done = torch.tensor([False, True]); terminated = torch.tensor([False, False])
    advantages, _ = gae(rewards, values, next_values, done, terminated, gamma=1., lam=1.)
    assert advantages[-1] == 2. and advantages[0] == 2.
    state = LagrangeState.create(2, [0., 1.], learning_rate=1., maximum=2.)
    state.update([3., -1.])
    assert torch.allclose(state.lambdas, torch.tensor([2., 0.]))


def test_causal_observation_and_split_sampler():
    builder = CausalObservation()
    with pytest.raises(ValueError):
        builder.observe(np.zeros(20))
    builder.append(np.zeros(18))
    assert builder.observe(np.zeros(20)).shape == (200,)
    sampler = StratifiedSampler(3)
    train, dev, test = sampler.generate("train", 20), sampler.generate("dev", 20), sampler.generate("test", 20)
    sampler.assert_disjoint(train, dev, test)
    assert [s.seed for s in sampler.generate("train", 3)] == [s.seed for s in sampler.generate("train", 3)]
