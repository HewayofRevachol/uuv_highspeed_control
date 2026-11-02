"""Check isolated run planning and config-to-CLI translation."""

import unittest
from pathlib import Path

from experiments.run import command_args, prepare


ROOT = Path(__file__).resolve().parents[2]


class ExperimentRunnerTest(unittest.TestCase):
    def test_false_boolean_uses_negative_switch(self):
        self.assertEqual(command_args({"fsp_enabled": False, "no_show": True}),
                         ["--no-fsp-enabled", "--no-show"])

    def test_dry_run_keeps_runs_separate(self):
        config = ROOT / "experiments/configs/ddat_cpu_smoke.json"
        _, first_dir, first_command = prepare(config, "controller_a", dry_run=True)
        _, second_dir, second_command = prepare(config, "controller_b", dry_run=True)
        self.assertNotEqual(first_dir, second_dir)
        self.assertIn(str(first_dir), first_command)
        self.assertIn(str(second_dir), second_command)
        self.assertFalse(first_dir.exists())


if __name__ == "__main__":
    unittest.main()
