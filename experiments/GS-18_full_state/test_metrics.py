import json
import unittest
from pathlib import Path

import numpy as np

from experiments.gs12_joint_route.joint_reference import joint_task_reference
from metrics import DT, N_STEPS, full_reference, inertial_velocity, score


TASKS = Path(__file__).with_name("tasks_dev.jsonl")


class TestFullStateMetrics(unittest.TestCase):
    def test_reference_kinematics(self):
        for line in TASKS.read_text(encoding="utf-8").splitlines():
            task = json.loads(line)
            full = full_reference(task)
            old = joint_task_reference(task, np.arange(N_STEPS+1)*DT)
            np.testing.assert_allclose(full[:, :5], old[:, :5], atol=1e-12)
            np.testing.assert_allclose(inertial_velocity(full), old[:, 5:8], atol=1e-12)
            np.testing.assert_allclose(full[:, 8], np.gradient(full[:, 3], DT, edge_order=2))
            np.testing.assert_allclose(full[:, 9], np.cos(full[:, 3]) *
                                       np.gradient(np.unwrap(full[:, 4]), DT, edge_order=2))

    def test_each_state_contributes_to_score(self):
        task = json.loads(TASKS.read_text(encoding="utf-8").splitlines()[0])
        reference = full_reference(task)
        self.assertAlmostEqual(score(reference, reference, "complete_valid")["J10"], 0.)
        for i in range(10):
            measured = reference.copy()
            measured[1:, i] += 0.01
            self.assertGreater(score(measured, reference, "complete_valid")["J10"], 0.)
        self.assertIsNone(score(reference, reference, "early_violation")["J10"])


if __name__ == "__main__":
    unittest.main()
