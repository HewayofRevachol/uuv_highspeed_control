import unittest

from beta_only import analyze_development, analyze_validation, method_stats


def row(method, task, value, **changes):
    return {"method": method, "task_id": task, "J10": value,
            "status": "complete_valid", "qp_failures": 0,
            "command_saturation_cycles": 0, "actual_saturation_steps": 0,
            "rmse": {"E": value}, "groups": {"position": value}, **changes}


class TestBetaOnlySelection(unittest.TestCase):
    def test_paired_fixed_oracle_and_rl_comparison(self):
        methods = ["beta_0.55", "beta_0.575", "RL_beta"]
        rows = [row(method, task, value) for method, values in
                zip(methods, ((2., 4.), (3., 1.), (1., 2.)))
                for task, value in zip(("a", "b"), values)]
        result = analyze_development(rows, methods, ["a", "b"])
        self.assertEqual(result["best_fixed"], "beta_0.575")
        self.assertAlmostEqual(result["taskwise_fixed_oracle"]["mean_J10"], 1.5)
        self.assertAlmostEqual(result["taskwise_fixed_oracle"]["reduction_fraction"], .25)
        self.assertAlmostEqual(result["RL_vs_best_fixed"]["reduction_fraction"], .25)
        self.assertEqual(result["RL_vs_best_fixed"]["wins"], 1)

    def test_failure_or_saturation_cannot_win_by_lower_complete_only_mean(self):
        methods = ["beta_0.55", "beta_0.575", "beta_0.6", "RL_beta"]
        rows = [row(method, task, 1.) for method in methods for task in ("a", "b")]
        rows[2].update(J10=None, status="qp_failure", qp_failures=1)
        rows[3]["J10"] = .001
        rows[4].update(J10=.001, command_saturation_cycles=1)
        rows[5]["J10"] = .001
        result = analyze_development(rows, methods, ["a", "b"])
        self.assertEqual(result["best_fixed"], "beta_0.55")
        self.assertFalse(result["methods"]["beta_0.575"]["eligible"])
        self.assertFalse(result["methods"]["beta_0.6"]["eligible"])

    def test_missing_or_duplicate_pair_is_rejected(self):
        for rows in ([row("beta_0.55", "a", 1.)],
                     [row("beta_0.55", "a", 1.), row("beta_0.55", "a", 2.)]):
            with self.assertRaisesRegex(ValueError, "exactly one row"):
                method_stats(rows, ["beta_0.55"], ["a", "b"])

    def test_validation_keeps_candidate_and_requires_quality(self):
        methods = ["beta_0.55", "beta_0.575", "RL_beta"]
        rows = [row(method, "a", value) for method, value in zip(methods, (1., .9, .1))]
        result = analyze_validation(rows, methods, ["a"], "beta_0.575", .05)
        self.assertEqual(result["candidate_method"], "beta_0.575")
        self.assertAlmostEqual(result["candidate_vs_baseline"]["reduction_fraction"], .1)
        self.assertTrue(result["accepted"])
        rows[1]["actual_saturation_steps"] = 1
        self.assertFalse(analyze_validation(rows, methods, ["a"], "beta_0.575", .05)["accepted"])


if __name__ == "__main__":
    unittest.main()
