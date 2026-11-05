import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

from experiments.h15_cat_mpc import evaluate
from experiments.h15_cat_mpc.evaluate import _validate_methods
from experiments.h15_cat_mpc.method_contract import (policy_execution_metadata,
    validate_method_checkpoint)


class H15EvaluateContractTests(unittest.TestCase):
    def test_fixed_gate_cannot_use_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "fixed method A0"):
            validate_method_checkpoint("A0", Path("ckpt_0000.pt"))
        with self.assertRaisesRegex(ValueError, "fixed methods"):
            _validate_methods(["A0"], [Path("ckpt_0000.pt")])

    def test_policy_requires_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_method_checkpoint("PPO", None)
        with self.assertRaisesRegex(ValueError, "requires"):
            _validate_methods(["PPO"], [])

    def test_unknown_and_mismatched_methods_fail(self):
        with self.assertRaisesRegex(ValueError, "unknown method"):
            validate_method_checkpoint("PPOish", None)
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_method_checkpoint("PPO_0100", Path("ckpt_0200.pt"))

    def test_multiple_checkpoints_are_real_files(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "ckpt_0000.pt"
            checkpoint.write_bytes(b"checkpoint")
            _validate_methods(["PPO"], [checkpoint])

    def test_execution_metadata_matches_action_source(self):
        fixed = policy_execution_metadata("A4", None, deterministic=True)
        self.assertEqual(fixed["method_family"], "fixed_gate")
        self.assertEqual(fixed["action_generation"], "fixed_gate")
        self.assertFalse(fixed["policy_deterministic"])
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "ckpt_0000.pt"
            checkpoint.write_bytes(b"checkpoint")
            sampled = policy_execution_metadata("PPO", checkpoint, deterministic=False)
            self.assertEqual(sampled["action_generation"], "policy_gaussian_sample")
            self.assertFalse(sampled["policy_deterministic"])
            mean = policy_execution_metadata("PPO", checkpoint, deterministic=True)
            self.assertEqual(mean["action_generation"], "policy_mean")
            self.assertTrue(mean["policy_deterministic"])


@pytest.fixture
def evaluation_cli(tmp_path, monkeypatch):
    """Run the actual CLI planner while replacing the expensive worker pool."""
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    path = tmp_path / "geometry.csv"
    path.write_text("x,y\n0,0\n1,0\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = [{"id": "geometry", "path": path.name, "path_sha256": digest}]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"t_fit_manifest": "t_fit.json",
        "train_manifest": "old_train.json", "dev_manifest": "dev.json",
        "holdout_manifest": "holdout.json"}))
    for name, split, seeds in (("t_fit", "T_fit", [51001]),
                               ("old_train", "train", [31001]),
                               ("dev", "dev", [32001])):
        (tmp_path / f"{name}.json").write_text(json.dumps(
            {"split": split, "cases": cases, "seeds": seeds}))
    (tmp_path / "holdout.json").write_text(json.dumps({
        "splits": {"holdout": [{"family": "geometry", "path": path.name}]},
        "experiment_freeze": {"seeds": [102061]}}))
    jobs = []
    pool_entries = []

    class FakePool:
        def __init__(self, **kwargs):
            pool_entries.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, function, planned):
            jobs.extend(planned)
            return [{"method": job["method"], "seed": job["seed"],
                     "success": True, "fallback_periods": 0} for job in planned]

    monkeypatch.setattr(evaluate.concurrent.futures, "ProcessPoolExecutor", FakePool)

    def run(extra=(), split="train", output="out"):
        out = tmp_path / output
        monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(config),
            "--split", split, "--out", str(out), *map(str, extra)])
        evaluate.main()
        return json.loads((out / "run.json").read_text())

    return run, tmp_path, jobs, pool_entries, digest


def test_cli_train_uses_t_fit_and_records_all_path_hashes(evaluation_cli):
    run, root, jobs, pools, digest = evaluation_cli
    record = run(["--methods", "A0"])
    assert record["manifest"] == str(root / "t_fit.json")
    assert [job["seed"] for job in jobs] == [51001]
    assert record["case_path_sha256"] == {"geometry.csv": digest}
    assert record["deterministic_policy"] is False
    assert len(pools) == 1


@pytest.mark.parametrize("stochastic", [False, True])
def test_cli_policy_mode_and_checkpoint_hash_match_jobs(evaluation_cli, stochastic):
    run, root, jobs, pools, _ = evaluation_cli
    checkpoint = root / "ckpt_0000.pt"
    checkpoint.write_bytes(b"policy bytes")
    extra = ["--methods", "PPO", "--policy-checkpoint", checkpoint]
    if stochastic:
        extra.append("--stochastic-policy")
    record = run(extra)
    assert record["deterministic_policy"] is (not stochastic)
    assert all(job["policy_deterministic"] is (not stochastic) for job in jobs)
    assert record["method_contract"]["policy_mode"] == (
        "gaussian_sample" if stochastic else "mean")
    assert record["policy_checkpoint_sha256"][str(checkpoint)] == hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    assert len(pools) == 1


def test_cli_checkpoint_list_retains_checkpoint_label_for_one_file(evaluation_cli):
    run, root, jobs, _, _ = evaluation_cli
    checkpoint = root / "ckpt_0000.pt"
    checkpoint.write_bytes(b"policy bytes")
    run(["--methods", "PPO", "--policy-checkpoints", checkpoint])
    assert [job["method"] for job in jobs] == ["PPO_0000"]


@pytest.mark.parametrize("combination", ["fixed_with_checkpoint", "ppo_without_checkpoint",
                                        "unknown_method", "both_checkpoint_options"])
def test_cli_invalid_checkpoint_combinations_leave_no_output(evaluation_cli, combination):
    run, root, jobs, pools, _ = evaluation_cli
    checkpoint = root / "ckpt_0000.pt"
    checkpoint.write_bytes(b"policy bytes")
    extra = {"fixed_with_checkpoint": ["--policy-checkpoint", checkpoint],
        "ppo_without_checkpoint": ["--methods", "PPO"],
        "unknown_method": ["--methods", "mystery"],
        "both_checkpoint_options": ["--methods", "PPO", "--policy-checkpoint", checkpoint,
                                    "--policy-checkpoints", checkpoint]}[combination]
    with pytest.raises((ValueError, SystemExit)):
        run(extra)
    assert not (root / "out").exists()
    assert not pools and not jobs


def test_cli_legacy_holdout_records_actual_hash(evaluation_cli):
    run, _, jobs, _, digest = evaluation_cli
    record = run(["--methods", "A0"], split="holdout")
    assert record["case_path_sha256"] == {"geometry.csv": digest}
    assert [job["seed"] for job in jobs] == [102061]


def test_cli_holdout_supplied_hash_must_match(evaluation_cli):
    run, root, jobs, pools, _ = evaluation_cli
    path = root / "holdout.json"
    manifest = json.loads(path.read_text())
    manifest["splits"]["holdout"][0]["path_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash mismatch"):
        run(["--methods", "A0"], split="holdout")
    assert not (root / "out").exists()
    assert not pools and not jobs


@pytest.mark.parametrize("invalid_field", ["missing_normalization", "split",
                                          "compensation_mode", "label_version",
                                          "normalization_sha256"])
def test_cli_predictor_contract_fails_before_output(evaluation_cli, invalid_field):
    run, root, jobs, pools, _ = evaluation_cli
    normalization = root / "normalization.json"
    normalization.write_text(json.dumps({"split": "train", "compensation_mode": "on"}))
    predictor = root / "predictor.json"
    manifest = {"split": "train", "compensation_mode": "on",
        "label_version": evaluate.RESIDUAL_LABEL_VERSION,
        "normalization_sha256": hashlib.sha256(normalization.read_bytes()).hexdigest()}
    if invalid_field != "missing_normalization":
        manifest[invalid_field] = "invalid"
    predictor.write_text(json.dumps({"manifest": manifest}))
    extra = ["--methods", "A0", "--predictor", predictor]
    if invalid_field != "missing_normalization":
        extra += ["--adapter-normalization", normalization]
    with pytest.raises(ValueError):
        run(extra)
    assert not (root / "out").exists()
    assert not pools and not jobs


def test_cli_matching_predictor_reaches_pool(evaluation_cli):
    run, root, jobs, pools, _ = evaluation_cli
    normalization = root / "normalization.json"
    normalization.write_text(json.dumps({"split": "train", "compensation_mode": "on"}))
    predictor = root / "predictor.json"
    predictor.write_text(json.dumps({"manifest": {"split": "train",
        "compensation_mode": "on", "label_version": evaluate.RESIDUAL_LABEL_VERSION,
        "normalization_sha256": hashlib.sha256(normalization.read_bytes()).hexdigest()}}))
    run(["--methods", "A0", "--predictor", predictor,
         "--adapter-normalization", normalization])
    assert len(pools) == 1
    assert jobs[0]["predictor_checkpoint"] == predictor


if __name__ == "__main__":
    unittest.main()
