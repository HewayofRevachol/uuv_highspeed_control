"""Check selected component behavior and reject incompatible plugin contracts."""

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from evaluation.ddat import run_episode
from research.controllers.ddat import ExperimentDDATPathEnv
from research.variants import (controller_class, create_environment, create_network,
                               model_class, network_class)
from simulator.batch_mpc import BatchedUUVModel
from simulator.batch_path_env import PathBank
from training.ddat import evaluate_paths


ROOT = Path(__file__).resolve().parents[2]
SMALL_ENV = {
    "horizon": 2, "admm_iterations": 2, "seed": 19,
    "randomize_plant": False, "current_max_mps": 0.,
    "attitude_pitch_weight": 0., "attitude_yaw_weight": 0.,
    "fsp_enabled": False, "preview_mode": "dense", "yaw_reward": "heading",
}


@pytest.fixture
def bank(tmp_path):
    source = ROOT / "DDAT_trajectories/val/constant_turn_01.csv"
    path = tmp_path / "short_path.csv"
    path.write_text("\n".join(source.read_text().splitlines()[:5]) + "\n")
    return PathBank([path], "cpu")


def install_module(monkeypatch, name, **symbols):
    module = ModuleType(name)
    module.__dict__.update(symbols)
    monkeypatch.setitem(sys.modules, name, module)
    return name


def test_default_factory_preserves_seeded_reset_and_step(bank):
    factory = create_environment(SMALL_ENV, bank, 1, "cpu")
    direct = ExperimentDDATPathEnv(bank, batch_size=1, device="cpu", **SMALL_ENV)
    ids = torch.zeros(1, dtype=torch.long)
    torch.testing.assert_close(factory.reset(ids), direct.reset(ids), rtol=0, atol=0)
    for first, second in zip(factory.step(torch.zeros(1, 3))[:3],
                             direct.step(torch.zeros(1, 3))[:3]):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_controller_options_change_training_and_evaluation_behavior(bank, monkeypatch):
    class RewardController(ExperimentDDATPathEnv):
        def __init__(self, *args, reward_bonus, **kwargs):
            self.reward_bonus = reward_bonus
            super().__init__(*args, **kwargs)

        def step(self, action):
            observation, reward, done, info = super().step(action)
            return observation, reward + self.reward_bonus, done, info

    module = install_module(monkeypatch, "research.test_reward_controller",
                            ExperimentDDATPathEnv=RewardController)
    config = {**SMALL_ENV, "controller_module": module,
              "controller_kwargs": {"reward_bonus": 2.5}}
    _, baseline = run_episode(bank, None, SMALL_ENV, "cpu", 19, False, "baseline")
    _, changed = run_episode(bank, None, config, "cpu", 19, False, "custom")
    expected = baseline["return"] + 2.5 * baseline["steps"]
    assert changed["return"] == pytest.approx(expected, abs=1e-5)
    args = SimpleNamespace(**config, device="cpu", eval_envs=1)
    training_rows = evaluate_paths(None, bank, args, 19)
    assert training_rows[0]["return"] == pytest.approx(expected, abs=1e-5)


def test_configured_model_is_used_for_prediction_and_plant(bank, monkeypatch):
    class FasterDynamics(BatchedUUVModel):
        def derivative(self, *args, **kwargs):
            result = super().derivative(*args, **kwargs)
            result[..., 5] += .3
            return result

    module = install_module(monkeypatch, "research.test_faster_model",
                            DynamicsModel=FasterDynamics)
    config = {**SMALL_ENV, "model_module": module}
    selected = create_environment(config, bank, 1, "cpu")
    default = create_environment(SMALL_ENV, bank, 1, "cpu")
    assert model_class(config) is FasterDynamics
    assert isinstance(selected.model, FasterDynamics)
    assert selected.mpc.model is selected.model
    for env in (selected, default):
        env.reset(torch.zeros(1, dtype=torch.long))
        env.step(torch.zeros(1, 3))
    assert not torch.allclose(selected.state, default.state)


@pytest.mark.parametrize("resolver,key", [
    (controller_class, "controller_module"),
    (network_class, "network_module"),
    (model_class, "model_module"),
])
def test_component_rejects_module_outside_research(resolver, key):
    with pytest.raises(ValueError, match=r"research\.\*"):
        resolver({key: "os"})


@pytest.mark.parametrize("symbol,error", [
    (None, "must define ActorCritic"),
    (lambda: None, "must be a class"),
    (type("IncompleteNetwork", (), {}), "missing methods"),
])
def test_incompatible_network_fails_before_training(monkeypatch, symbol, error):
    symbols = {} if symbol is None else {"ActorCritic": symbol}
    module = install_module(monkeypatch, "research.test_invalid_network", **symbols)
    with pytest.raises(ValueError, match=error):
        network_class({"network_module": module})


@pytest.mark.parametrize("kwargs", [[], {1: "invalid"}, {"backend": "osqp"},
                                     {"model_factory": "invalid"}])
def test_invalid_controller_options_are_rejected(bank, kwargs):
    with pytest.raises(ValueError, match="controller_kwargs"):
        create_environment({**SMALL_ENV, "controller_kwargs": kwargs}, bank, 1, "cpu")


def test_network_cannot_override_observation_dimension():
    with pytest.raises(ValueError, match="managed arguments: observation_dim"):
        create_network({"network_kwargs": {"observation_dim": 1}}, 112, "cpu")


def test_unknown_environment_override_is_rejected(bank):
    with pytest.raises(ValueError, match="unknown environment overrides: horizn"):
        create_environment(SMALL_ENV, bank, 1, "cpu", horizn=2)
