from experiments.h15_cat_mpc import e2_matrix
from experiments.h15_cat_mpc.residual_predictor import ResidualAR1


def test_predictor_gate_requires_same_three_components_in_both_geometries(monkeypatch):
    monkeypatch.setattr(e2_matrix, "offline_forecast_metrics",
                        lambda row, predictor: {"25": {"count": 60, "mae": [1.] * 5},
                                                 "5": {"count": 60, "mae": [1.] * 5}})
    def fake_mae(rows, predictor, horizon):
        if predictor.version.startswith("persistence"):
            values = [1., 1., 1., 1., 1.]
        else:
            values = [.9, .9, .9, 1., 1.]
        return {name: {"mae": values, "count": 60, "per_episode_counts": [60]}
                for name in ("dev_reversal", "dev_coupled")}

    monkeypatch.setattr(e2_matrix, "_mae_by_geometry", fake_mae)
    result = e2_matrix.select_predictor([{}], ResidualAR1(version="train_ar1"))
    assert result["selected"] == "ar_or_persistence"
    assert result["passing_horizon"] == 5
    assert result["by_horizon"]["5"]["common_improving_components"] == [0, 1, 2]


def test_predictor_gate_falls_back_when_ar_degrades_a_component(monkeypatch):
    monkeypatch.setattr(e2_matrix, "offline_forecast_metrics",
                        lambda row, predictor: {"25": {"count": 60, "mae": [1.] * 5},
                                                 "5": {"count": 60, "mae": [1.] * 5}})
    def fake_mae(rows, predictor, horizon):
        if predictor.version.startswith("persistence"):
            values = [1., 1., 1., 1., 1.]
        else:
            values = [.9, .9, .9, .9, 1.1]
        return {name: {"mae": values, "count": 60, "per_episode_counts": [60]}
                for name in ("dev_reversal", "dev_coupled")}

    monkeypatch.setattr(e2_matrix, "_mae_by_geometry", fake_mae)
    result = e2_matrix.select_predictor([{}], ResidualAR1(version="train_ar1"))
    assert result["selected"] == "persistence"
    assert result["ar_passed"] is False
