import numpy as np
import torch

from aav_mpc.h15_challenge_paths import PROFILES, generate
from aav_mpc.controller import EndpointExtendedCondensedEnv
from aav_mpc.diagnose import predict_accepted_step
from aav_mpc.h15_experiment import summarize
from simulator.batch_path_env import PathBank
from simulator.model import LoopPath


def test_challenge_geometry_and_splits(tmp_path):
    from aav_mpc.h15_challenge_paths import write_corpus

    manifest = write_corpus(tmp_path)
    assert len(manifest["splits"]) == 3
    for split, entries in manifest["splits"].items():
        assert {p["family"] for p in entries} == set(PROFILES[split])
        for entry in entries:
            path = LoopPath(entry["path"])
            assert abs(path.length - 300.) < .1
            assert np.all(path.speed == 15.)
            assert np.all(np.diff(path.distances) > 0)
            assert abs(path.yaw[0]) < 1e-4
    params = [PROFILES[split]["tight_turn"]["radius_m"] for split in PROFILES]
    assert len(set(params)) == 3
    tight, _ = generate("tight_turn", PROFILES["development"]["tight_turn"])
    assert np.max(abs(tight[:, 1])) > 20.


def test_paired_summary_excludes_failed_rmse():
    def row(method, good, rmse):
        return {"split": "development", "family": "turn", "seed": 1,
                "method": method, "strict_success": good, "simulator_success": good,
                "first_anomaly": None if good else [1., "speed"],
                "control_period_p95_ms": 20., "control_period_p99_ms": 25.,
                "fallback_count": 0, "steps": 10, "position_rmse_m": rmse}

    result = summarize([row("osqp25", False, 100.),
                        row("endpoint40", True, 1.)], ["osqp25", "endpoint40"])
    paired = result["paired_vs_osqp25"]["endpoint40"]
    assert paired["strict_success_delta"] == 1
    assert paired["common_success_pairs"] == 0
    assert paired["common_success_position_rmse_delta_m"] is None


def test_accepted_command_prediction_matches_identified_plant_step():
    path_file = "data/datasets/h15_mpc_replication/replication_coupled_a.csv"
    bank = PathBank([path_file], "cpu")
    bank.speed[:] = 15.
    bank.first_state[:, 5] = 15.
    bank.durations[:] = bank.lengths / 15.
    env = EndpointExtendedCondensedEnv(
        bank, batch_size=1, device="cpu", horizon=20, backend="rti_cpu",
        seed=1, randomize_plant=False, current_max_mps=0.,
        current_profile="steady", admm_iterations=400, admm_rho=10.,
        attitude_pitch_weight=100., attitude_yaw_weight=30.,
        fsp_enabled=False, preview_mode="multiscale", yaw_reward="course")
    env.reset(torch.zeros(1, dtype=torch.long))
    env.disturbance_std.zero_()
    before = env.state[0].clone()
    actuator = env.actuator[0].clone()
    queue = env.command_history[0].clone()
    command = env.base_command[0].clone()
    predicted = predict_accepted_step(env.mpc, before, actuator, queue, command)
    env.step(torch.zeros(1, 3))
    torch.testing.assert_close(predicted, env.state[0], rtol=0., atol=1e-12)


def test_fix_holdout_is_frozen_before_evaluation(tmp_path):
    from aav_mpc.h15_fix_holdout_paths import SEEDS, write_corpus

    manifest = write_corpus(tmp_path)
    entries = manifest["splits"]["holdout"]
    assert len(entries) == 4
    assert tuple(manifest["experiment_freeze"]["seeds"]) == SEEDS
    assert manifest["experiment_freeze"]["robust_prediction"] == {
        "fin_gain_scale": .85, "fin_tau_scale": 2.,}
    for entry in entries:
        path = LoopPath(entry["path"])
        assert abs(path.length - 300.) < .1
        assert np.all(path.speed == 15.)
