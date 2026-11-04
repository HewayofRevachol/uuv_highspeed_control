import math
import unittest

from experiments.h15_mainline.observation_features import FEATURE_NAMES, from_period_row


class ObservationFeaturesTests(unittest.TestCase):
    def test_hidden_parameters_do_not_enter_observation(self):
        row = {"lateral_error_m": "1", "depth_error_m": "2",
               "speed_error_mps": "-0.5", "pitch_deg": "3",
               "thrust_actual_n": "750000", "elevator_actual_deg": "10",
               "rudder_actual_deg": "-5", "command_json": "[900000, 0.1, -0.2]",
               "queue_before_json": "[[600000, 0.2, -0.3]]"}
        baseline = from_period_row(row)
        altered = from_period_row(row | {"true_fin_gain": 100,
                                         "future_disturbance": [999],
                                         "oracle_label": "unsafe"})
        self.assertEqual(baseline, altered)
        self.assertEqual(len(baseline), len(FEATURE_NAMES))
        self.assertAlmostEqual(baseline[4], .5)
        self.assertAlmostEqual(baseline[5], math.radians(10))

    def test_nonfinite_sensor_value_rejected(self):
        row = {"lateral_error_m": "nan", "depth_error_m": "0",
               "speed_error_mps": "0", "pitch_deg": "0",
               "thrust_actual_n": "0", "elevator_actual_deg": "0",
               "rudder_actual_deg": "0", "command_json": "[0, 0, 0]",
               "queue_before_json": "[[0, 0, 0]]"}
        with self.assertRaises(ValueError):
            from_period_row(row)


if __name__ == "__main__":
    unittest.main()
