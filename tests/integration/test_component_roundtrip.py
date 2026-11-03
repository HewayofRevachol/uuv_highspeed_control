"""Small real PPO runs preserve component configuration through checkpoint reload."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from evaluation.ddat import controller_config, load_actor, run_episode
from experiments.run import command_args
from research.variants import create_environment, create_network
from simulator.batch_path_env import PathBank


ROOT = Path(__file__).resolve().parents[2]
NETWORK_OPTIONS = {"hidden_sizes": [32, 32], "encoder_sizes": [32, 16]}
CONTROLLER_OPTIONS = {"current_profile": "harmonic"}


@pytest.mark.parametrize("entrypoint", [["DDAT_train.py"], ["-m", "training.ddat"]],
                         ids=["legacy", "module"])
def test_tiny_training_checkpoint_roundtrip(tmp_path, entrypoint):
    source = ROOT / "DDAT_trajectories/val/constant_turn_01.csv"
    short_csv = "\n".join(source.read_text().splitlines()[:5]) + "\n"
    path_dir = tmp_path / "paths"
    for split in ("train", "val"):
        (path_dir / split).mkdir(parents=True)
        (path_dir / split / "short.csv").write_text(short_csv)
    output = tmp_path / "training"
    arguments = {
        "device": "cpu", "envs": 1, "eval_envs": 1,
        "total_steps": 2, "rollout": 2, "horizon": 2, "admm_iterations": 2,
        "epochs": 1, "minibatch": 2, "torch_threads": 1,
        "randomize_plant": False, "fsp_enabled": False,
        "path_dir": str(path_dir), "output": str(output),
        "network_kwargs": NETWORK_OPTIONS, "controller_kwargs": CONTROLLER_OPTIONS,
    }
    result = subprocess.run([sys.executable, *entrypoint, *command_args(arguments)],
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint_path = output / "final_actor.pt"
    actor, checkpoint = load_actor(checkpoint_path, "cpu")
    config = controller_config(checkpoint["config"])
    assert checkpoint["training_step"] == 2
    assert config["network_kwargs"] == NETWORK_OPTIONS
    assert config["controller_kwargs"] == CONTROLLER_OPTIONS
    assert config["model_module"] == "research.models.uuv"
    assert json.loads((output / "config.json").read_text())["network_kwargs"] == NETWORK_OPTIONS
    assert sum(p.numel() for p in actor.parameters()) < sum(
        p.numel() for p in create_network({}, checkpoint["observation_dim"], "cpu").parameters())
    restored = create_network(config, checkpoint["observation_dim"], "cpu")
    restored.load_state_dict(checkpoint["agent"])
    observations = torch.randn(2, checkpoint["observation_dim"])
    torch.testing.assert_close(actor.deterministic(observations),
                               restored.deterministic(observations), rtol=0, atol=0)
    bank = PathBank([path_dir / "val/short.csv"], "cpu")
    env = create_environment(config, bank, 1, "cpu")
    assert env.current_profile == "harmonic"
    rows, summary = run_episode(bank, actor, config, "cpu", 19, False, "reloaded")
    assert 1 <= len(rows) <= 4
    assert summary["steps"] == len(rows)
    assert torch.isfinite(torch.tensor([row["reward"] for row in rows])).all()


@pytest.mark.parametrize("entrypoint", [
    ["DDAT_train.py"], ["-m", "training.ddat"],
    ["validate_ddat.py"], ["-m", "evaluation.ddat"],
])
def test_legacy_and_module_cli_help(entrypoint):
    result = subprocess.run([sys.executable, *entrypoint, "--help"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--device" in result.stdout


def test_legacy_imports_alias_new_modules():
    import DDAT_train
    import validate_ddat
    from training import ddat as training
    from evaluation import ddat as evaluation

    assert DDAT_train is training
    assert validate_ddat is evaluation
