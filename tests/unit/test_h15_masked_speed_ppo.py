import unittest
import json
from types import SimpleNamespace

import numpy as np
import torch

from experiments.h15_mainline.masked_speed_ppo import (GRID, OBSERVATION_SIZE,
                                                        MaskedSpeedActorCritic,
                                                        decision_observation)


class MaskedSpeedPpoTests(unittest.TestCase):
    def test_observation_includes_queue_residual_and_candidate_status(self):
        class Path:
            distances = np.asarray([0., 100.])

            def tangent_at(self, progress):
                return np.asarray([1., 0., 0.])

            def position_at(self, progress):
                return np.asarray([progress, 0., 0.])

        class Dynamics:
            def evaluate(self, state, command):
                predicted = state.copy()
                return predicted, None, None

        state = [0., 0., 0., 0., 0., 15., 0., 0., 0., 0.]
        previous = {"state_before_json": json.dumps(state),
                    "state_after_json": json.dumps(state[:8] + [.01, .02]),
                    "actuator_before_json": "[0, 0, 0]",
                    "queue_before_json": json.dumps([[0., .01, 0.]] * 5),
                    "command_json": "[0, 0, 0]", "thrust_actual_n": 0.,
                    "elevator_actual_deg": 0., "rudder_actual_deg": 0.}
        candidates = [{"candidate_mps": speed, "physical_accepted": speed == 15.,
                       "qp_valid": True, "cost": 1.,
                       "scenarios": [{"risk": .5, "min_margins": {
                           "lateral_m": 4., "depth_m": 2.5, "pitch_deg": 20.,
                           "tangent_water_speed_error_mps": 1.}}]}
                      for speed in GRID]
        sample = {"lateral_error_m": 0., "depth_error_m": 0.,
                  "pitch_deg": 0., "speed_error_mps": 0.}
        estimate = SimpleNamespace(fin_gain=.85, fin_tau_scale=2.,
                                   gain_radius=.25, tau_radius=1.5)
        observation, mask = decision_observation(
            state, sample, [0., 0., 0.], [[0., .01, 0.]] * 5, Path(), 0.,
            estimate, candidates, 15., previous, Dynamics())
        self.assertEqual(observation.shape, (60,))
        self.assertAlmostEqual(observation[10], .01/(np.pi/4))
        self.assertAlmostEqual(observation[26], .1)
        self.assertEqual(observation[45:50].tolist(), [.5] * 5)
        self.assertEqual(observation[50:55].tolist(), [1.] * 5)
        self.assertEqual(mask.tolist(), [False, False, True, False, False])

    def test_policy_never_selects_rejected_intent(self):
        torch.manual_seed(7)
        policy = MaskedSpeedActorCritic()
        observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        mask = np.asarray([False, True, False, False, False])
        for _ in range(20):
            action, logp, value = policy.choose(observation, mask)
            self.assertEqual(GRID[action], 14.)
            self.assertTrue(np.isfinite([logp, value]).all())

    def test_empty_mask_is_rejected(self):
        policy = MaskedSpeedActorCritic()
        with self.assertRaises(ValueError):
            policy.choose(np.zeros(OBSERVATION_SIZE), np.zeros(5, dtype=bool))


if __name__ == "__main__":
    unittest.main()
