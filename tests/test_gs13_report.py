"""Pair completeness, failures and immutable reports must survive aggregation."""
import csv
import json

import pytest

from experiments.gs13_srppo.report import paired_summary, write_report


def row(scene, method, cost, *, capacity=False, failed=False, truncated=False):
    return {"scene_id": scene, "method": method, "split": "dev",
            "capacity_limited": capacity, "failed": failed, "truncated": truncated,
            "complete": not failed and not truncated, "cost": cost,
            "tracking_integral": cost * 2., "qp_failures": int(failed),
            "deadline_misses": 0, "saturation_fraction": .1}


def dataset():
    return [row("a", "fixed_best", 10.), row("b", "fixed_best", 20.),
            row("c", "fixed_best", 50., capacity=True),
            row("a", "rule_best", 8.), row("b", "rule_best", 16.),
            row("c", "rule_best", 40., capacity=True),
            row("a", "srppo", 6.), row("b", "srppo", 12.),
            row("c", "srppo", 100., capacity=True, failed=True)]


def test_capacity_strata_and_both_strong_baselines_are_exactly_paired():
    result = paired_summary(dataset())
    feasible = result["strata"]["feasible"]
    assert feasible["methods"]["srppo"]["n"] == 2
    assert feasible["methods"]["srppo"]["mean_cost"] == 9.
    assert feasible["paired"]["fixed_best"]["srppo"]["gain_fraction"] == pytest.approx(.4)
    assert feasible["paired"]["rule_best"]["srppo"]["gain_fraction"] == pytest.approx(.25)
    assert result["strata"]["all"]["methods"]["srppo"]["failures"] == 1
    capacity = result["strata"]["capacity_limited"]["paired"]["fixed_best"]["srppo"]
    assert capacity["failure_delta"] == 1 and capacity["gain_fraction"] == -1.
    assert capacity["common_success"]["n"] == 0


def test_common_success_never_erases_failed_pairs_or_absorbing_cost():
    rows = [row("a", "fixed_best", 10.), row("b", "fixed_best", 10.),
            row("a", "srppo", 5.), row("b", "srppo", 200., failed=True)]
    result = paired_summary(rows)["strata"]["feasible"]
    paired = result["paired"]["fixed_best"]["srppo"]
    assert result["methods"]["srppo"]["mean_cost"] == 102.5
    assert paired["n"] == 2 and paired["failures"] == 1
    assert paired["gain_fraction"] < 0
    assert paired["common_success"]["gain_fraction"] == .5
    assert paired["common_success"]["excluded_pairs"] == 1


def test_truncation_retained_and_makes_comparison_ineligible():
    rows = [row("a", "fixed_best", 10.), row("a", "srppo", .1, truncated=True)]
    result = paired_summary(rows)["strata"]["feasible"]
    assert result["methods"]["srppo"]["truncated"] == 1
    assert result["methods"]["srppo"]["n"] == 1
    assert not result["methods"]["srppo"]["comparison_eligible"]
    paired = result["paired"]["fixed_best"]["srppo"]
    assert not paired["comparison_eligible"] and paired["gain_fraction"] is None
    assert paired["truncated_pairs"] == 1


def test_duplicate_and_missing_pairs_fail_closed():
    with pytest.raises(ValueError, match="duplicate"):
        paired_summary(dataset() + [dataset()[0]])
    with pytest.raises(ValueError, match="missing paired"):
        paired_summary(dataset()[:-1])
    with pytest.raises(ValueError, match="baseline missing"):
        paired_summary([row("a", "srppo", 1.)])


@pytest.mark.parametrize("field,value", [("cost", float("nan")), ("cost", float("inf")),
                                        ("tracking_integral", -1.), ("qp_failures", .5),
                                        ("saturation_fraction", 1.1)])
def test_invalid_numeric_values_are_rejected(field, value):
    rows = dataset()
    rows[0][field] = value
    with pytest.raises(ValueError):
        paired_summary(rows)


def test_metadata_and_status_must_agree():
    rows = dataset()
    rows[-1]["capacity_limited"] = False
    with pytest.raises(ValueError, match="metadata differs"):
        paired_summary(rows)
    rows = dataset()
    rows[-1]["complete"] = True
    with pytest.raises(ValueError, match="exactly one"):
        paired_summary(rows)


def test_output_is_immutable_reproducible_and_preserves_all_rows(tmp_path):
    rows = dataset()
    first = write_report(rows, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    second = write_report(reversed(rows), tmp_path)
    assert first == second
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with (tmp_path / "paired_metrics.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == len(rows) and sum(r["failed"] == "True" for r in saved) == 1
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["strata"]["feasible"]["methods"]["srppo"]["n"] == 2
    assert "共同成功" in (tmp_path / "report.md").read_text()
    rows[-1]["cost"] += 1.
    with pytest.raises(FileExistsError, match="artifact differs"):
        write_report(rows, tmp_path)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_empty_stratum_and_zero_baseline_are_explicit():
    result = paired_summary([row("a", "fixed_best", 0.), row("a", "srppo", 0.)])
    empty = result["strata"]["capacity_limited"]["methods"]["srppo"]
    assert empty["n"] == 0 and empty["mean_cost"] is None and not empty["comparison_eligible"]
    assert result["strata"]["all"]["paired"]["fixed_best"]["srppo"]["gain_fraction"] is None
    with pytest.raises(ValueError, match="empty"):
        paired_summary([])
