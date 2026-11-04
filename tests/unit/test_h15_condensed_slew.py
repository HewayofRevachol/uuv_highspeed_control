import unittest

import numpy as np

from experiments.h15_mainline.run_offline_governor_smoke import ROOT, create_env


class CondensedSlewTests(unittest.TestCase):
    def test_slew_is_enforced_in_candidate_plan(self):
        path = ROOT / "data/datasets/h15_mainline_quickstart/train_turn_wide.csv"
        env = create_env(path, 31001, False)
        limits = np.array([1., .6, .4])
        env.mpc.set_command_slew_limits(limits)
        env.set_intent(15.)
        _, valid = env.mpc.solve(env.observed_augmented_state(), env._reference())
        self.assertTrue(bool(valid[0]))
        plan = env.mpc.plan[0].numpy()
        rates = np.diff(np.vstack((np.zeros(3), plan)), axis=0)
        self.assertTrue(np.all(np.abs(rates) <= limits + 1e-8))

    def test_invalid_slew_limits_are_rejected(self):
        path = ROOT / "data/datasets/h15_mainline_quickstart/train_turn_wide.csv"
        env = create_env(path, 31001, False)
        for limits in ([1., 0., .4], [1., np.nan, .4], [1., .6]):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                env.mpc.set_command_slew_limits(limits)

    def test_invalid_sqp_iterations_are_rejected(self):
        path = ROOT / "data/datasets/h15_mainline_quickstart/train_turn_wide.csv"
        env = create_env(path, 31001, False)
        for value in (0, 6, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                env.mpc.set_sqp_iterations(value)


if __name__ == "__main__":
    unittest.main()
