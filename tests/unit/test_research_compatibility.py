"""Guard compatibility with the checkpoint bundled in this research snapshot."""

import unittest
from pathlib import Path

import torch

from research.variants import controller_class, network_class


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "outputs/DDAT/run_20261029_114239/final_actor.pt"


class ResearchCompatibilityTest(unittest.TestCase):
    def test_bundled_checkpoint_loads_with_default_network(self):
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        agent = network_class(checkpoint["config"])(checkpoint["observation_dim"])
        agent.load_state_dict(checkpoint["agent"])
        self.assertEqual(agent.deterministic(torch.zeros(1, checkpoint["observation_dim"])).shape,
                         (1, 3))

    def test_legacy_config_resolves_default_controller(self):
        from research.controllers.ddat import ExperimentDDATPathEnv

        self.assertIs(controller_class({}), ExperimentDDATPathEnv)


if __name__ == "__main__":
    unittest.main()
