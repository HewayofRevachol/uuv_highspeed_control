import copy
import json
import sys

import pytest

from experiments.h15_cat_mpc import analyze_fixed_gates
from experiments.h15_cat_mpc.analyze_ppo import (P2_PROTOCOL, exploration_rules,
                                                summarize_pairs)


def episode(rmse=.1, physical=True, fallback=0, deadline=True, success=None):
    success = physical if success is None else success
    no_fallback = bool(physical and fallback == 0)
    return {"schema": 2, "command_schema": "issued_v2", "path": "geometry.csv",
        "seed": 1, "method": "A0", "success": success, "physical_strict": physical,
        "no_fallback_physical": no_fallback, "deadline_clean": deadline,
        "strict_mainline": no_fallback and deadline, "strict_hist": physical,
        "fallback_periods": fallback, "periods": 10,
        "trace": [{"command": [0., 0., 0.]}],
        "strict_substeps": [{"lateral_error_m": 8 * rmse, "depth_error_m": 0.,
            "speed_error_mps": 0., "pitch_rad": 0., "period": 0, "substep": 0,
            "time_s": 0., "physical_flags": {"lateral": not physical}}],
        "residual_forecast_errors": {str(h): {"count": 0, "ar1_mae": None,
            "persistence_mae": None} for h in (1, 5, 25)}}


def paired(row, count=2):
    return {("geometry", seed): copy.deepcopy(row) for seed in range(count)}


def test_audit_fallback_finishes_are_not_unintervened_improvements():
    summary = summarize_pairs(paired(episode(fallback=1)),
                              paired(episode(physical=False, fallback=1)))
    assert summary["physical_improved"] == 2
    assert summary["no_fallback_improved"] == 0
    assert summary["improved"] == 0
    assert summary["safety_precondition"] is True
    assert exploration_rules(summary)["rule1"] is False


def test_audit_lower_rmse_with_new_fallback_cannot_pass_precision_rule():
    summary = summarize_pairs(paired(episode(rmse=.05, fallback=1)),
                              paired(episode(rmse=.1)))
    assert summary["physical_worsened"] == 0
    assert summary["no_fallback_worsened"] == 2
    assert summary["common_no_fallback_successes"] == 0
    assert summary["normalized_rmse_relative_change"] is None
    assert summary["safety_precondition"] is False
    assert not any(exploration_rules(summary)[key] for key in ("rule1", "rule2", "rule3"))


def test_median_uses_individual_paired_relative_changes():
    baseline = {("g", i): episode(rmse=value) for i, value in enumerate([.01, .1, 1.])}
    candidate = {("g", i): episode(rmse=value) for i, value in enumerate([.02, .07, .85])}
    summary = summarize_pairs(candidate, baseline)
    assert summary["normalized_rmse_relative_change"] == pytest.approx(-.15)
    assert summary["candidate_normalized_rmse_median"] / summary["baseline_normalized_rmse_median"] - 1 == pytest.approx(-.3)


def test_fallback_improvements_cannot_cancel_a_paired_increase():
    baseline = {("g", 0): episode(fallback=2), ("g", 1): episode(fallback=2),
                ("g", 2): episode()}
    candidate = {("g", 0): episode(fallback=3), ("g", 1): episode(fallback=0),
                 ("g", 2): episode()}
    summary = summarize_pairs(candidate, baseline)
    assert summary["candidate_fallback_periods"] < summary["baseline_fallback_periods"]
    assert summary["paired_fallback_increases"] == 1
    assert summary["safety_precondition"] is False
    assert exploration_rules(summary)["rule3"] is False


def test_physical_regression_blocks_rules_even_when_both_had_fallback():
    baseline = {("g", 0): episode(fallback=1), ("g", 1): episode()}
    candidate = {("g", 0): episode(physical=False, fallback=1),
                 ("g", 1): episode(rmse=.05)}
    summary = summarize_pairs(candidate, baseline)
    assert summary["no_fallback_worsened"] == 0
    assert summary["physical_worsened"] == 1
    assert exploration_rules(summary)["rule2"] is False


def test_deadline_changes_remain_separate_from_physical_and_no_fallback():
    summary = summarize_pairs(paired(episode(deadline=False)), paired(episode()))
    assert summary["physical_worsened"] == 0
    assert summary["no_fallback_worsened"] == 0
    assert summary["deadline_worsened"] == 2
    assert summary["safety_precondition"] is True


def test_legacy_p2_thresholds_still_accept_real_clean_improvements():
    completion = summarize_pairs(paired(episode()), paired(episode(physical=False)))
    precision = summarize_pairs(paired(episode(rmse=.08)), paired(episode(rmse=.1)))
    assert exploration_rules(completion)["rule1"] is True
    assert exploration_rules(precision)["rule2"] is True
    assert P2_PROTOCOL["version"] == "legacy_p2_audit_repaired_v2"
    assert "not E4/E5" in P2_PROTOCOL["scope"]


def test_zero_baseline_error_cannot_hide_a_regression():
    baseline = {("g", 0): episode(rmse=0.), ("g", 1): episode(rmse=.1)}
    candidate = {("g", 0): episode(rmse=.01), ("g", 1): episode(rmse=.01)}
    summary = summarize_pairs(candidate, baseline)
    assert summary["normalized_rmse_relative_change"] is None
    assert exploration_rules(summary)["rule2"] is False


def test_fixed_gate_cli_reports_fallback_regression_consistently(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for method in ("A0", "A1", "A2", "A3", "A4"):
        directory = runs / method
        directory.mkdir(parents=True)
        row = episode(rmse=.1 if method == "A0" else .05,
                      fallback=0 if method == "A0" else 1)
        row["method"] = method
        (directory / "episode.json").write_text(json.dumps(row))
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["analyze_fixed_gates", "--runs", str(runs), "--out", str(out)])
    analyze_fixed_gates.main()
    report = json.loads(out.read_text())
    comparison = report["paired_vs_A0"]["A4"]
    assert comparison["no_fallback_worsened"] == 1
    assert comparison["physical_strict_worsened"] == 0
    assert comparison["common_no_fallback_successes"] == 0
    assert comparison["normalized_rmse_relative_change"] is None
    assert comparison["safety_precondition"] is False
    assert report["residual_prediction_mae"]["25"]["ar1_mae"] is None
