"""Check isolated run planning and config-to-CLI translation."""

import unittest
from pathlib import Path

from experiments.run import command_args, prepare, source_hashes


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

    def test_component_options_remain_one_json_cli_argument(self):
        import json

        options = {"hidden_sizes": [32, 16], "encoder_sizes": [16, 8]}
        arguments = command_args({"network_kwargs": options})
        self.assertEqual(arguments[0], "--network-kwargs")
        self.assertEqual(json.loads(arguments[1]), options)
        with self.assertRaisesRegex(ValueError, "JSON object"):
            command_args({"network_kwargs": []})

    def test_manifest_hashes_include_shared_implementations(self):
        hashes = source_hashes()
        for path in ("training/ddat.py", "training/ppo.py", "evaluation/ddat.py",
                     "evaluation/metrics.py", "research/models/uuv.py", "experiments/run.py"):
            self.assertIn(path, hashes)


if __name__ == "__main__":
    unittest.main()
