"""Exercise a selected dynamics model through plant and predictor integration."""

from pathlib import Path
import unittest

import torch

from research.models.uuv import DynamicsModel
from simulator.batch_mpc import BatchedUUVModel
from simulator.batch_path_env import BatchedPathResidualEnv, PathBank
from simulator.ddat_env import DDATPathEnv


ROOT = Path(__file__).resolve().parents[2]


class PerturbedDynamics(BatchedUUVModel):
    def derivative(self, *args, **kwargs):
        result = super().derivative(*args, **kwargs)
        result[..., 5] += .3
        return result


class ModelInjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = PathBank([ROOT / "DDAT_trajectories/val/constant_turn_01.csv"],
                            "cpu")

    def environment(self, **kwargs):
        return DDATPathEnv(self.bank, batch_size=1, device="cpu", horizon=2,
                           admm_iterations=4, randomize_plant=False,
                           current_max_mps=0., seed=12, **kwargs)

    def test_explicit_default_preserves_reset_and_step(self):
        default = self.environment(fsp_enabled=True)
        selected = self.environment(fsp_enabled=True, model_factory=DynamicsModel)
        ids = torch.zeros(1, dtype=torch.long)
        torch.testing.assert_close(default.reset(ids), selected.reset(ids),
                                   rtol=0, atol=0)
        action = torch.tensor([[.1, -.2, .3]])
        expected = default.step(action)
        actual = selected.step(action)
        for first, second in zip(expected[:3], actual[:3]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(default.state, selected.state, rtol=0, atol=0)

    def test_selected_dynamics_reaches_plant_mpc_and_fsp(self):
        default = self.environment(fsp_enabled=True)
        selected = self.environment(fsp_enabled=True, model_factory=PerturbedDynamics)
        self.assertIsInstance(selected.model, PerturbedDynamics)
        self.assertIs(selected.mpc.model, selected.model)
        self.assertIs(selected.fsp.model, selected.model)
        ids = torch.zeros(1, dtype=torch.long)
        default.reset(ids)
        observation = selected.reset(ids)
        self.assertEqual(observation.shape, (1, DDATPathEnv.observation_size))
        action = torch.zeros(1, 3)
        default.step(action)
        observation, reward, _, _ = selected.step(action)
        self.assertTrue(torch.isfinite(observation).all())
        self.assertTrue(torch.isfinite(reward).all())
        self.assertGreater((selected.state-default.state).abs().max().item(), 1e-6)

    def test_base_environment_accepts_factory(self):
        env = BatchedPathResidualEnv(self.bank, None, batch_size=1, device="cpu",
                                     horizon=2, model_factory=PerturbedDynamics)
        self.assertIsInstance(env.model, PerturbedDynamics)

    def test_custom_model_rejects_fixed_symbolic_rti(self):
        for backend in ("rti_cpu", "rti_cuda"):
            with self.subTest(backend=backend), self.assertRaisesRegex(
                    ValueError, "fixed symbolic UUV model"):
                self.environment(backend=backend, model_factory=PerturbedDynamics)

    def test_noncallable_factory_has_clear_error(self):
        with self.assertRaisesRegex(TypeError, "model_factory must be callable"):
            self.environment(model_factory=object())


if __name__ == "__main__":
    unittest.main()
