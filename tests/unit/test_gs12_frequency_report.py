"""Focused regressions for GS-12 frequency probes and paired decisions."""

import json

import numpy as np
import pytest

from experiments.gs12_gate import frequency, report


def test_frequency_case_preserves_nominal_delay_and_time_constants():
    case = frequency._case("probe", 6, [1., 0., 0.], .006, 80.)

    assert case["plant"] == {
        "mass": 1., "damping": 1., "thrust": 1., "fin": .9,
        "tau": 1., "delay": 3, "order": "second_all",
    }


def test_nonexcited_spectrum_preserves_mean_square_energy(monkeypatch):
    t = np.arange(640)*.1
    omega = 2*np.pi*6/64
    error = np.zeros((len(t), 3))
    error[:, 0] = np.sin(2*np.pi*8/64*t)
    signal = np.zeros_like(error)
    signal[:, 0] = .006*np.cos(omega*t)
    monkeypatch.setattr(frequency, "_load_error",
                        lambda *args: (t, error, signal, True))

    metrics = frequency._analysis("synthetic", None, omega,
                                  [1., 0., 0.], .006, 0.)

    assert metrics["nonexcited_energy"] == pytest.approx(.5)
    assert metrics["nonexcited_spectrum"][8] == pytest.approx(.5)
    assert metrics["amplitude_ratio"] == pytest.approx(1.)
    assert metrics["valid"]


def test_decision_does_not_reward_earlier_failure(tmp_path):
    (tmp_path/"selected_baselines.json").write_text(json.dumps({
        "fixed_best": "fixed0", "rule_best": "rule0",
        "scores": {"fixed0": [1, 20.], "rule0": [1, 21.]},
    }), encoding="utf-8")
    methods = ["fixed_best", "rule_best", "ppo"]

    def row(case_id, method, cost, complete):
        return {"case_id": case_id, "method": method,
                "capacity_limited": False, "tracking_cost_integral": cost,
                "complete": complete}

    # Equal cost rates but different failure times must not imply RL improves.
    rows = [row("failed", method, cost, False)
            for method, cost in zip(methods, [20., 21., 5.])]
    decision = report._decision(rows, methods, tmp_path, "pilot")
    assert "共同成功可行场景 0/1" in decision
    assert "不能判定 RL 增益或晋级" in decision
    assert "这是晋级信号" not in decision

    # Failed trajectories remain in failure counts, but not full cost means.
    rows.extend(row("success", method, 10., True) for method in methods)
    decision = report._decision(rows, methods, tmp_path, "pilot")
    assert "共同成功可行场景 1/2" in decision
    assert "fixed_best=1，ppo=1" in decision
    assert "代价变化 0.0%" in decision
    assert "这是晋级信号" not in decision
