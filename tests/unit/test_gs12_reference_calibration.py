import numpy as np
import pytest

from experiments.gs12_gate import run
from experiments.gs12_gate.plant_harness import Reference, make_case


@pytest.mark.parametrize("maneuver", ["speed", "turn", "depth", "coupled"])
@pytest.mark.parametrize("phase", [0., 1., 4.2])
def test_reference_velocity_is_position_derivative(maneuver, phase):
    case = make_case("reference", "train", 15, "nominal", maneuver, 1)
    case["phase"] = phase
    reference = Reference(case)
    t, h = 2., 1e-5
    velocity = (reference.at(t+h)[:3]-reference.at(t-h)[:3])/(2*h)
    np.testing.assert_allclose(velocity, reference.at(t)[5:8], atol=1e-8)
    np.testing.assert_allclose(reference.at(0)[:3], 0., atol=1e-12)


def test_calibration_replenishment_does_not_require_svd_direction(tmp_path, monkeypatch):
    cases = [make_case("cal0", "train_calibration", 15, "nominal", "straight", 1),
             make_case("cal1", "train_calibration", 15, "nominal", "straight", 2),
             make_case("pending_svd", "train", 15, "nominal", "frequency_direction", 3),
             make_case("replacement", "train", 15, "nominal", "straight", 4)]
    cases[2]["complex_direction"] = None
    monkeypatch.setattr(run, "_load_run", lambda _: ({}, cases))
    visited = []

    def fake_episode(case, method, path, **kwargs):
        assert case["maneuver"] != "frequency_direction"
        visited.append(case["case_id"])
        path.mkdir(parents=True)
        np.savez(path/"control.npz", t=[0.], d_raw=np.zeros((1, 5)),
                 errors=np.zeros((1, 4)), actual=np.zeros((1, 3)),
                 issued=np.zeros((1, 3)), queue=np.zeros((1, 5, 3)),
                 state=np.zeros((1, 10)))
        np.savez(path/"decisions.npz", observation=np.zeros((1, 200)))
        return {"case_id": case["case_id"], "complete": case["case_id"] != "cal0"}

    monkeypatch.setattr(run, "run_episode", fake_episode)
    result = run.collect(tmp_path, episodes=2)
    assert visited == ["cal0", "cal1", "replacement"]
    assert result["episodes"] == 3
