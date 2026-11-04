import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.h15_mainline.candidate_filter import replay_plan, screen_candidate, SCALE
from simulator.model import LoopPath
from simulator.rti_symbolic import ActuatorAwareCpuDynamics


class CandidateReplayTests(unittest.TestCase):
    def test_candidate_settles_previous_intent_before_switching(self):
        class FakeMpc:
            def __init__(self):
                self.plan = torch.zeros(1, 35, 3)
                self.previous = torch.zeros(1, 3)
                self.states = torch.zeros(1, 36, 22)
                self.last_qp_valid = torch.ones(1, dtype=torch.bool)
                self.last_primal_residual = torch.zeros(1)
                self.last_dual_residual = torch.zeros(1)
                self.last_status = ["not_solved"]

            def solve(self, state, reference):
                self.last_status[0] = "failed"
                self.last_qp_valid[0] = False
                return None, self.last_qp_valid

        class FakeEnv:
            def __init__(self):
                self.mpc = FakeMpc()
                self.intent_mps = 14.
                self._reference_step = 0
                self._reference_value = 15.
                self.steps = torch.tensor([1])

            def reference_rate(self):
                if self._reference_step < int(self.steps[0]):
                    self._reference_value += max(
                        -.05, min(.05, self.intent_mps-self._reference_value))
                    self._reference_step += 1
                return self._reference_value

            def set_intent(self, value):
                self.intent_mps = value

            def _reference(self):
                self.reference_rate()
                return torch.zeros(1, 35, 8)

            def observed_augmented_state(self):
                return torch.zeros(1, 22)

        env = FakeEnv()
        estimate = SimpleNamespace(constraint_radii={})
        result = screen_candidate(env, None, 17., estimate, {}, {})
        self.assertEqual(result["first_ramped_mps"], 15.)
        self.assertEqual((env.intent_mps, env._reference_step, env._reference_value),
                         (14., 1, 14.95))

    def test_candidate_probe_restores_solver_and_reference_state(self):
        class FakeMpc:
            def __init__(self):
                self.plan = torch.zeros(1, 35, 3)
                self.previous = torch.zeros(1, 3)
                self.states = torch.zeros(1, 36, 22)
                self.last_qp_valid = torch.ones(1, dtype=torch.bool)
                self.last_primal_residual = torch.zeros(1)
                self.last_dual_residual = torch.zeros(1)
                self.last_status = ["previous"]

            def solve(self, state, reference):
                self.plan.fill_(1)
                self.previous.fill_(2)
                self.states.fill_(3)
                self.last_qp_valid.fill_(False)
                self.last_primal_residual.fill_(4)
                self.last_dual_residual.fill_(5)
                self.last_status[0] = "failed"
                return None, self.last_qp_valid

        class FakeEnv:
            def __init__(self):
                self.mpc = FakeMpc()
                self.intent_mps = 15.
                self._reference_step = 7
                self._reference_value = 14.8

            def set_intent(self, value):
                self.intent_mps = value

            def reference_rate(self):
                return self._reference_value

            def _reference(self):
                self._reference_step = 8
                self._reference_value = 14.85
                return torch.zeros(1, 35, 8)

            def observed_augmented_state(self):
                return torch.zeros(1, 22)

        env = FakeEnv()
        estimate = SimpleNamespace(constraint_radii={})
        result = screen_candidate(env, None, 14., estimate, {}, {})
        self.assertEqual(result["first_rejection"], "qp_invalid")
        self.assertEqual(result["qp_status"], "failed")
        self.assertEqual((env.intent_mps, env._reference_step, env._reference_value),
                         (15., 7, 14.8))
        self.assertEqual(env.mpc.last_status, ["previous"])
        self.assertTrue(bool(env.mpc.last_qp_valid[0]))
        for tensor in (env.mpc.plan, env.mpc.previous, env.mpc.states,
                       env.mpc.last_primal_residual, env.mpc.last_dual_residual):
            self.assertTrue(bool(torch.all(tensor == 0)))

    def test_initial_physical_and_queue_violations_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path_file = Path(directory) / "path.csv"
            with path_file.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("north_m", "east_m", "depth_m",
                                 "target_tangent_speed_mps"))
                writer.writerows(((0., 0., 0., 15.), (100., 0., 0., 15.)))
            path = LoopPath(path_file)
            state = np.array([0., 0., 0., np.deg2rad(41), 0.,
                              15., 0., 0., 0., 0.])
            fields = ("lateral_m", "depth_m", "pitch_deg",
                      "tangent_water_speed_error_mps")
            config = {"horizon_periods": 35,
                      "strict_limits": dict(zip(fields, (8., 5., 40., 2.))),
                      "terminal_extra_margin": {f: 0. for f in fields}}
            radii = {str(h): {f: 0. for f in
                     ("lateral_m", "depth_m", "pitch_deg",
                      "tangent_water_speed_mps")} for h in (1, 5, 25, 35)}
            zero = np.zeros(3)
            queue = np.zeros((5, 3))
            result = replay_plan(state, zero, queue, [zero]*35, 0., path,
                                 .85, 2., radii, config)
            self.assertEqual((result["first_rejection"], result["first_time_s"]),
                             ("pitch_deg", 0.))
            queue[0, 0] = 2_000_000.
            result = replay_plan(state, zero, queue, [zero]*35, 0., path,
                                 .85, 2., radii, config)
            self.assertEqual((result["first_rejection"], result["first_time_s"]),
                             ("queue_box", 0.))

    def test_first_step_matches_independent_symbolic_augmented_dynamics(self):
        with tempfile.TemporaryDirectory() as directory:
            path_file = Path(directory) / "path.csv"
            with path_file.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("north_m", "east_m", "depth_m",
                                 "target_tangent_speed_mps"))
                writer.writerows(((0., 0., 0., 15.), (100., 0., 0., 15.)))
            path = LoopPath(path_file)
            state = np.array([0., 0., 0., 0., 0., 15., 0., 0., 0., 0.])
            actual = np.array([200000., .04, -.03])
            queue = np.array([[250000., .05, -.04],
                              [200000., .03, -.02],
                              [150000., .01, 0.],
                              [100000., 0., 0.],
                              [50000., 0., 0.]])
            command = np.array([300000., .06, -.05])
            radii = {str(h): {field: 0. for field in
                     ("lateral_m", "depth_m", "pitch_deg",
                      "tangent_water_speed_mps")} for h in (1, 5, 25, 35)}
            config = {"horizon_periods": 35,
                      "strict_limits": {field: 1e9 for field in
                      ("lateral_m", "depth_m", "pitch_deg",
                       "tangent_water_speed_error_mps")},
                      "terminal_extra_margin": {field: 0. for field in
                      ("lateral_m", "depth_m", "pitch_deg",
                       "tangent_water_speed_error_mps")}}
            result = replay_plan(state, actual, queue, [command]*35, 0., path,
                                 .85, 2., radii, config)
            normalized = np.r_[state, actual/SCALE,
                               (queue[:3]/SCALE).ravel()]
            expected = ActuatorAwareCpuDynamics(.1, .85, 2.).evaluate(
                normalized[None, None], (command/SCALE)[None, None])[0][0, 0]
            observed = np.r_[result["first_step_state"],
                             np.asarray(result["first_step_actuator"])/SCALE,
                             (np.asarray(result["first_step_queue"])/SCALE).ravel()]
            np.testing.assert_allclose(observed, expected, rtol=0., atol=1e-12)


if __name__ == "__main__":
    unittest.main()
