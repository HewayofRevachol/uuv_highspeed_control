from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.gs12_gate import episode
from experiments.gs12_gate.features import FRAME_SCALE
from experiments.gs12_gate.model_bridge import HybridDynamics, THETA_SAFE
from experiments.gs12_gate.plant_harness import make_case
from simulator.batch_mpc import CONTROL_SCALE


@pytest.fixture
def cheap_mpc(monkeypatch):
    """Keep the real plant/model and episode timing, without solving a QP."""
    class FixedMPC:
        def __init__(self, model):
            self.model = model
            self.dynamics = HybridDynamics()
            self.plan = model.tensor(np.zeros((1, 35, 3)))
            self.previous = model.tensor(np.zeros((1, 3)))
            self.last_status = ["solved"]
            self.last_primal_residual = model.tensor([0.])
            self.last_dual_residual = model.tensor([0.])

        def solve(self, state, reference, **kwargs):
            command = self.model.tensor([.72, .01, -.015])
            self.plan[:] = command
            self.previous[:] = command
            return command[None]*self.model.control_scale, torch.tensor([True])

    monkeypatch.setattr(episode, "HybridMPC", FixedMPC)


def test_locked_model_keeps_updating_measured_thrust_rate(tmp_path, monkeypatch, cheap_mpc):
    observers = []
    original = episode.ThrustRateObserver

    class TrackingObserver(original):
        def __init__(self):
            super().__init__()
            self.measurements = []
            observers.append(self)

        def update(self, value, dt=.1):
            self.measurements.append(float(value))
            return super().update(value, dt)

    monkeypatch.setattr(episode, "ThrustRateObserver", TrackingObserver)
    case = make_case("observer", "test", 15., "nominal", "straight", 7, .2)
    pre = tmp_path/"pre"
    snapshot = pre/"snapshot.pt"
    episode.run_episode(case, "hybrid_zero", pre, snapshot_out=snapshot)
    saved_rate = torch.load(snapshot, weights_only=False)["observer_rate"]
    locked = tmp_path/"locked"
    episode.run_episode(case, "hybrid_zero", locked, snapshot_path=snapshot,
                        freeze_gate=True)
    # Initialization plus one update for each of the two completed periods.
    assert len(observers[-1].measurements) == 3
    assert observers[-1].rate != pytest.approx(saved_rate)
    with np.load(locked/"control.npz") as control:
        assert control["model_state"][1, 13] != pytest.approx(saved_rate)
        np.testing.assert_array_equal(control["theta_used"][0], control["theta_used"][1])
        np.testing.assert_array_equal(control["d_used"][0], control["d_used"][1])


def test_truncation_bootstraps_from_the_same_features_as_continuation(
        tmp_path, monkeypatch, cheap_mpc):
    estimators = []
    estimator_defaults = episode.SlidingWindowResponseEstimator.DEFAULTS

    class ObservedEstimator:
        DEFAULTS = estimator_defaults

        def __init__(self, normalization, parameters):
            self.history = deque(maxlen=10)
            self.last = None
            estimators.append(self)

        def update(self, row):
            self.history.append(row)
            n = len(self.history)
            self.last = SimpleNamespace(fin_gain=.85+.01*n, fin_tau_scale=2.-.1*n,
                gain_radius=.2+.01*n, tau_radius=1.+.1*n,
                source="E1", fallback_reason=None)
            return self.last

    class FeatureValuePolicy(torch.nn.Module):
        variant, action_dim = "standard", 4

        def value(self, observation):
            return observation[:, 180:184].sum(-1)+observation[:, 175:180].sum(-1)

        def act(self, observation, deterministic=False):
            return {"target": torch.zeros((len(observation), 4)),
                    "raw_z": torch.zeros((len(observation), 4)),
                    "log_prob": torch.zeros(len(observation)),
                    "value": self.value(observation)}

    monkeypatch.setattr(episode, "SlidingWindowResponseEstimator", ObservedEstimator)
    case = make_case("bootstrap", "test", 15., "nominal", "straight", 8, .3)
    policy = FeatureValuePolicy()
    short, full = tmp_path/"short", tmp_path/"full"
    episode.run_episode(case, "ppo_mean", short, policy=policy,
                        max_seconds=.2, gate_phase_steps=2)
    assert len(estimators[0].history) == 2  # Includes the last completed transition.
    episode.run_episode(case, "ppo_mean", full, policy=policy, gate_phase_steps=2)
    with np.load(short/"decisions.npz") as cut, np.load(full/"decisions.npz") as continued:
        assert cut["truncated"][-1] and not cut["terminal"][-1]
        expected = continued["observation"][1]  # The actual next decision at t=.2.
        actual = cut["next_observation"][-1]
        assert np.linalg.norm(expected[175:180]) > 0.
        np.testing.assert_allclose(actual[175:184], expected[175:184], rtol=1e-6)
        assert cut["next_value"][-1] == pytest.approx(continued["value"][1])


def test_rule_residual_scores_use_applied_model_and_channel_scales():
    applied_offset = np.array([1., 2., 3., .4, .5])

    class ParameterDynamics:
        def evaluate(self, state, command, theta):
            result = np.zeros((1, 1, 25))
            result[0, 0, 5:10] = (theta[0]-THETA_SAFE[0])*applied_offset
            return result, None, None

    used = THETA_SAFE+np.array([1., 0.])
    estimated = THETA_SAFE+np.array([2., 0.])
    residual = np.array([2., 3., 4., 5., 6.])
    observed = applied_offset+.1*residual
    base, corrected = episode._shadow_errors(ParameterDynamics(), np.zeros(25),
        np.zeros(3), observed, estimated, used, residual)
    # Each residual group exactly explains its own applied-model error.
    np.testing.assert_allclose(corrected[1:], 0., atol=1e-14)
    normalized = .1*residual/FRAME_SCALE[-5:]
    np.testing.assert_allclose(base[1:], [abs(normalized[0]),
        np.linalg.norm(normalized[[2, 3]]), np.linalg.norm(normalized[[1, 4]])])
    assert base[0] == pytest.approx(np.linalg.norm(observed/FRAME_SCALE[-5:]))
    assert corrected[0] == pytest.approx(
        np.linalg.norm((observed-2*applied_offset)/FRAME_SCALE[-5:]))
