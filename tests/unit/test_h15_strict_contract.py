import unittest

from experiments.h15_mainline.strict_contract import evaluate


class StrictContractTests(unittest.TestCase):
    def setUp(self):
        self.clean = {"time_s": .1, "lateral_error_m": 0.,
                      "depth_error_m": 0., "pitch_deg": 0.,
                      "speed_error_mps": 0.}
        self.flags = dict(completed=True, period_samples=[self.clean],
                          substep_samples=[self.clean], mpc_fallback=False,
                          candidate_empty=False, backup_used=False,
                          deadline_missed=False, fallback_defined=True)

    def test_substep_violation_preserves_historical_success(self):
        result = evaluate(**(self.flags | {"substep_samples": [self.clean | {
            "time_s": .02, "pitch_deg": 41.}]}))
        self.assertTrue(result["strict_hist"])
        self.assertFalse(result["strict_mainline"])
        self.assertEqual(result["first_substep_violation"], (.02, "pitch_deg"))

    def test_controller_events_fail_mainline(self):
        for field, value in (("candidate_empty", True), ("backup_used", True),
                             ("deadline_missed", True), ("fallback_defined", False)):
            with self.subTest(field=field):
                result = evaluate(**(self.flags | {field: value}))
                self.assertFalse(result["strict_mainline"])

    def test_unknown_events_cannot_pass(self):
        result = evaluate(**(self.flags | {"candidate_empty": None}))
        self.assertIsNone(result["strict_mainline"])

    def test_backup_completion_requires_physical_limits(self):
        result = evaluate(**(self.flags | {"backup_used": True}))
        self.assertTrue(result["completed_with_backup"])
        failed = evaluate(**(self.flags | {"backup_used": True,
            "substep_samples": [self.clean | {"speed_error_mps": 2.1}]}))
        self.assertFalse(failed["completed_with_backup"])


if __name__ == "__main__":
    unittest.main()
