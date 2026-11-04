import copy
import tempfile
import unittest
from pathlib import Path

from experiments.h15_mainline.cycle_log import FIELDS, TIMING_FIELDS, validate
from experiments.h15_mainline.cycle_writer import timing_summary, write_once


class CycleLogTests(unittest.TestCase):
    def diagnostic(self):
        return dict(zip(FIELDS, (1, "reference_chain_smoke", "case", 0, .1,
                                 [0.] * 10, [0.] * 10, [0.] * 3, [0.] * 3,
                                 [[0.] * 3 for _ in range(5)], [0.] * 3,
                                 None, 15., 15., None, None, None, None,
                                 None, "solved",
                                 dict(zip(TIMING_FIELDS,
                                          (None, None, None, 20., 8., 30.))))))

    def test_diagnostic_keeps_candidate_events_unknown(self):
        self.assertIsNone(validate(self.diagnostic())["candidate_empty"])
        false_claim = copy.deepcopy(self.diagnostic())
        false_claim["candidate_empty"] = False
        with self.assertRaises(ValueError):
            validate(false_claim)

    def test_formal_empty_set_requires_fallback(self):
        record = self.diagnostic()
        record["method"] = "B0_shared"
        record["candidate_empty"] = True
        record["candidate_rejections"] = [
            {"candidate_mps": 15., "reason": "screen_budget"}]
        record["timing_ms"] = dict(zip(TIMING_FIELDS, (1., 3., 0., 20., 8., 33.)))
        with self.assertRaises(ValueError):
            validate(record)
        record["fallback"] = "task_abort"
        self.assertTrue(validate(record)["candidate_empty"])

    def test_fixed_intent_rejection_can_fallback_when_other_candidates_exist(self):
        record = self.diagnostic()
        record["method"] = "B0_shared"
        record["candidate_empty"] = False
        record["candidate_rejections"] = [
            {"candidate_mps": 15., "reason": "speed"},
            {"candidate_mps": 14., "reason": None}]
        record["timing_ms"] = dict(zip(TIMING_FIELDS, (1., 3., 0., 20., 8., 33.)))
        record["fallback"] = "task_abort"
        self.assertIsNone(validate(record)["selected_candidate_mps"])

    def test_formal_timing_aggregation_and_consecutive_writer(self):
        rows = []
        for period in range(3):
            row = self.diagnostic()
            row["method"] = "B0_shared"
            row["period"] = period
            row["time_s"] = .1 * (period + 1)
            row["candidate_empty"] = False
            row["candidate_rejections"] = [
                {"candidate_mps": 15., "reason": None}]
            row["selected_candidate_mps"] = 15.
            row["timing_ms"] = dict.fromkeys(TIMING_FIELDS, float(period + 1))
            rows.append(row)
        summary = timing_summary(rows)
        self.assertEqual(summary["estimator"]["p50"], 2.)
        self.assertEqual(summary["candidate_screen"]["max"], 3.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycles.jsonl"
            write_once(path, rows)
            write_once(path, rows)
            with self.assertRaises(ValueError):
                write_once(path, rows[1:])


if __name__ == "__main__":
    unittest.main()
