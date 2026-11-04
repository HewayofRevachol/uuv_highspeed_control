import unittest

import torch

from experiments.h15_mainline.progress_reference import ProgressReferenceEnv
from simulator.batch_path_env import PathBank
from pathlib import Path


class ProgressReferenceTests(unittest.TestCase):
    def make_reference(self):
        env = object.__new__(ProgressReferenceEnv)
        env.steps = torch.zeros(1, dtype=torch.long)
        return env

    def test_dynamic_intent_ramps_from_previous_reference(self):
        env = self.make_reference()
        env.set_intent(14.)
        values = [env.reference_rate()]
        for step in range(1, 22):
            env.steps[0] = step
            values.append(env.reference_rate())
        self.assertAlmostEqual(values[-1], 14.)
        env.set_intent(17.)
        for step in range(22, 30):
            env.steps[0] = step
            values.append(env.reference_rate())
        self.assertTrue(all(abs(b-a) <= .050000001 for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[22], 14.05)

    def test_constant_intent_and_reset(self):
        env = self.make_reference()
        env.set_intent(14.)
        for step in range(25):
            env.steps[0] = step
            self.assertAlmostEqual(env.reference_rate(), max(14., 15.-.05*step))
        env.steps[0] = 0
        self.assertEqual(env.reference_rate(), 15.)

    def test_nonfinite_intent_rejected(self):
        env = self.make_reference()
        with self.assertRaises(ValueError):
            env.set_intent(float("nan"))

    def test_reference_nodes_follow_switching_ramp(self):
        root = Path(__file__).resolve().parents[2]
        bank = PathBank([root / "data/datasets/h15_mainline_quickstart/train_turn_wide.csv"], "cpu")
        env = ProgressReferenceEnv(bank, batch_size=1, device="cpu", horizon=5,
                                   backend="rti_cpu", seed=31001, fsp_enabled=False)
        env.reset(torch.zeros(1, dtype=torch.long))
        env.set_intent(14.)
        for _ in range(22):
            env.step(torch.zeros(1, 3))
        before = env.reference_rate()
        env.set_intent(17.)
        self.assertAlmostEqual(env.reference_rate(), before)
        nodes = env._reference()
        position, _, _, _, _ = env.paths.sample(
            env.path_ids, env.progress[:, None] + .1 * (before + .05))
        # The first target progresses by one 0.1 s step at no more than 0.05 m/s
        # above the current reference, even after a large intent switch.
        self.assertLessEqual(abs(env.reference_rate() - before), .050000001)
        self.assertEqual(nodes.shape, (1, 5, 8))
        torch.testing.assert_close(nodes[0, 0, :3],
                                   position[0, 0] - .1 * env.current_estimate[0])
        env.step(torch.zeros(1, 3))
        self.assertAlmostEqual(env.reference_rate(), before + .05)


if __name__ == "__main__":
    unittest.main()
