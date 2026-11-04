import json
import unittest

from experiments.h15_mainline.observation_features import FEATURE_NAMES
from experiments.h15_mainline.response_estimator import SlidingWindowResponseEstimator


class ResponseEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.statistics = {"split": "train", "feature_names": FEATURE_NAMES,
                           "mean": [0.] * len(FEATURE_NAMES),
                           "scale": [100.] * len(FEATURE_NAMES)}
        self.calibration = {"method": "empirical_dev_replay_residual_quantile",
                            "horizons_control_periods": [1, 5, 25, 35],
                            "radii": {"1": {"pitch_deg": 1.},
                                      "5": {"pitch_deg": 2.},
                                      "25": {"pitch_deg": 3.},
                                      "35": {"pitch_deg": 4.}}}

    def row(self, step):
        state = [0., 0., 0., 0., 0., 15., 0., 0., 0., 0.]
        return {"time_s": str(.1 * (step + 1)),
                "state_before_json": json.dumps(state),
                "state_after_json": json.dumps(state),
                "actuator_before_json": "[0, 0, 0]",
                "queue_before_json": "[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]",
                "command_json": "[0, 0, 0]",
                "mpc_base_command_json": "[0, 0, 0]",
                "mpc_plan_commands_json": json.dumps([[0, 0, 0]] * 35),
                "thrust_actual_n": "0", "elevator_actual_deg": "0",
                "rudder_actual_deg": "0", "lateral_error_m": "0",
                "depth_error_m": "0", "speed_error_mps": "0",
                "pitch_deg": "0"}

    def test_short_and_low_excitation_use_e0_with_calibrated_state_radii(self):
        estimator = SlidingWindowResponseEstimator(self.statistics,
                                                   calibration=self.calibration)
        first = estimator.update(self.row(0))
        self.assertEqual((first.source, first.fallback_reason), ("E0", "short_history"))
        self.assertEqual(first.constraint_radii["25"]["pitch_deg"], 3.)
        for step in range(1, 10):
            last = estimator.update(self.row(step))
        self.assertEqual((last.source, last.fallback_reason),
                         ("E0", "low_gain_excitation"))
        self.assertGreaterEqual(last.gain_radius, .25)
        self.assertGreaterEqual(last.tau_radius, 1.5)

    def test_nonfinite_and_out_of_order_history_fail_closed(self):
        estimator = SlidingWindowResponseEstimator(self.statistics)
        estimator.update(self.row(0))
        broken = self.row(1) | {"pitch_deg": "nan"}
        result = estimator.update(broken)
        self.assertEqual(result.source, "E0")
        self.assertTrue(result.fallback_reason.startswith("invalid_history"))
        out_of_order = estimator.update(self.row(0))
        self.assertEqual(out_of_order.samples, 1)

    def test_ood_observation_uses_e0(self):
        estimator = SlidingWindowResponseEstimator(self.statistics)
        for step in range(7):
            estimator.update(self.row(step))
        result = estimator.update(self.row(7) | {"lateral_error_m": "801"})
        self.assertEqual((result.source, result.fallback_reason),
                         ("E0", "ood_observation"))

    def test_stale_history_uses_e0(self):
        estimator = SlidingWindowResponseEstimator(self.statistics)
        for step in range(7):
            estimator.update(self.row(step))
        result = estimator.update(self.row(7) | {"time_s": "2.0"})
        self.assertEqual((result.source, result.fallback_reason),
                         ("E0", "stale_history"))

    def test_full_plan_is_required_and_must_match_first_command(self):
        estimator = SlidingWindowResponseEstimator(self.statistics)
        missing = self.row(0)
        del missing["mpc_plan_commands_json"]
        self.assertEqual(estimator.update(missing).source, "E0")
        mismatched = self.row(1)
        plan = json.loads(mismatched["mpc_plan_commands_json"])
        plan[0][1] = .1
        mismatched["mpc_plan_commands_json"] = json.dumps(plan)
        result = estimator.update(mismatched)
        self.assertEqual((result.source, result.fallback_reason),
                         ("E0", "invalid_history:ValueError"))


if __name__ == "__main__":
    unittest.main()
