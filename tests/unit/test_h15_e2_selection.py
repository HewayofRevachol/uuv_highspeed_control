import json

import pytest

from experiments.h15_cat_mpc import e2_matrix
from experiments.h15_cat_mpc.residual_predictor import ResidualAR1


def _metrics(improvements, counts=60):
    ar = [1. - value for value in improvements]
    return {name: {"mae": ar, "count": counts, "per_episode_counts": [counts]}
            for name in ("geom_a", "geom_b")}


def test_predictor_gate_selects_ar_only_with_three_consistent_components(monkeypatch):
    def fake_metrics(rows, predictor, horizon):
        if predictor.version == "train_ar1":
            return _metrics([.06, .06, .06, 0., 0.])
        return _metrics([0.] * 5)

    monkeypatch.setattr(e2_matrix, "_mae_by_geometry", fake_metrics)
    result = e2_matrix.select_predictor([{}], ResidualAR1(version="train_ar1"))
    assert result["selected"] == "ar_or_persistence"
    assert result["ar_passed"] is True


def test_predictor_gate_falls_back_when_h25_coverage_is_insufficient(monkeypatch):
    def fake_metrics(rows, predictor, horizon):
        if predictor.version == "train_ar1":
            return _metrics([.2, .2, .2, .2, .2], counts=49 if horizon == 25 else 60)
        return _metrics([0.] * 5)

    monkeypatch.setattr(e2_matrix, "_mae_by_geometry", fake_metrics)
    result = e2_matrix.select_predictor([{}], ResidualAR1(version="train_ar1"))
    assert result["selected"] == "persistence"
    assert result["ar_passed"] is False


@pytest.mark.parametrize("last_error", [0.1, float("nan")])
def test_zero_error_baseline_and_nonfinite_ar_cannot_bypass_gate(monkeypatch, last_error):
    def fake_metrics(rows, predictor, horizon):
        values = ([.8, .8, .8, 1., last_error] if predictor.version == "train_ar1"
                  else [1., 1., 1., 1., 0.])
        return {name: {"mae": values, "count": 60, "per_episode_counts": [60]}
                for name in ("geom_a", "geom_b")}

    monkeypatch.setattr(e2_matrix, "_mae_by_geometry", fake_metrics)
    result = e2_matrix.select_predictor([{}], ResidualAR1(version="train_ar1"))
    assert result["selected"] == "persistence"
    # Selection artifacts must be valid JSON even for degenerate baselines.
    json.dumps(result, allow_nan=False)


def test_improving_components_must_agree_across_geometries(monkeypatch):
    def fake_metrics(rows, predictor, horizon):
        metrics = _metrics([.1, .1, .1, 0., 0.]) if predictor.version == "train_ar1" else _metrics([0.] * 5)
        if predictor.version == "train_ar1":
            metrics["geom_b"]["mae"] = [1., 1., .9, .9, .9]
        return metrics

    monkeypatch.setattr(e2_matrix, "_mae_by_geometry", fake_metrics)
    assert e2_matrix.select_predictor([{}], ResidualAR1(version="train_ar1"))["selected"] == "persistence"


def test_short_episode_cannot_be_hidden_by_geometry_aggregate(monkeypatch):
    def fake_metrics(rows, predictor, horizon):
        metrics = _metrics([.1] * 5) if predictor.version == "train_ar1" else _metrics([0.] * 5)
        for item in metrics.values():
            item.update(count=249, per_episode_counts=[49, 100, 100])
        return metrics

    monkeypatch.setattr(e2_matrix, "_mae_by_geometry", fake_metrics)
    result = e2_matrix.select_predictor([{}], ResidualAR1(version="train_ar1"))
    assert result["selected"] == "persistence"
