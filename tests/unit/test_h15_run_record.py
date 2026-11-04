import copy
import unittest

from experiments.h15_mainline.run_record import (TOP_LEVEL, TIMING_STAGES,
                                                  RESULT_FIELDS, validate)


class RunRecordTests(unittest.TestCase):
    def record(self):
        results = dict.fromkeys(RESULT_FIELDS)
        results.update(episodes=3, strict_hist_completed=3,
                       fallback_episodes=0, first_violations={})
        return dict(zip(TOP_LEVEL, (1, "h15_smoke", {"config": "a"*64},
                                    {"data": "b"*64}, [1, 2, 3],
                                    dict.fromkeys(TIMING_STAGES), results)))

    def test_missing_mainline_evidence_stays_unknown(self):
        record = self.record()
        self.assertIsNone(validate(record)["results"]["strict_mainline_completed"])
        claimed = copy.deepcopy(record)
        claimed["results"]["strict_mainline_completed"] = 3
        with self.assertRaises(ValueError):
            validate(claimed)

    def test_nonfinite_timing_rejected(self):
        record = self.record()
        record["timing_ms"]["mpc"] = {"p50": 1., "p95": 2.,
                                       "p99": float("nan"), "max": 3.}
        with self.assertRaises(ValueError):
            validate(record)


if __name__ == "__main__":
    unittest.main()
