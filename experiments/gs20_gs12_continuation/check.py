"""Minimal checks only: imports, references, synthetic RMSE and policy dimensions.

No plant construction, MPC solve, trajectory simulation, PPO update or training.
"""
import importlib
import json

import numpy as np
import torch

from .common import HERE, ROOT, sha
from .metrics import error, score
from .policy import ActorCritic
from .prepare import build_sets
from .references import reference


def main():
    for name in ('prepare', 'run', 'analyze', 'frequency', 'train', 'closed_loop', 'plot'):
        importlib.import_module(f'{__package__}.{name}')
    sets = build_sets()
    ids = [t['task_id'] for tasks in sets.values() for t in tasks]
    assert len(ids) == len(set(ids))
    for group in ('bridge', 'maneuvers', 'static_dev', 'switch_dev', 'switch_train', 'switch_test'):
        task = sets[group][-1]
        t, h = np.array([7.1, 15.3, 46.7, 49.3, 95.5, 98.3, 103.1]), 1e-5
        ref = reference(task, t)
        derivative = (reference(task, t+h)[:, :3]-reference(task, t-h)[:, :3])/(2*h)
        assert np.max(np.abs(derivative-ref[:, 5:])) < 3e-7
    # k=2 makes the ramp integral contain a zero-frequency term.
    task = dict(sets['static_dev'][0], axis='speed', k=2, phase_rad=.31)
    t = np.array([3.3, 15.3, 20.2])
    derivative = (reference(task, t+1e-5)[:, 0]-reference(task, t-1e-5)[:, 0])/2e-5
    np.testing.assert_allclose(derivative, reference(task, t)[:, 5], atol=3e-7, rtol=0)
    ref, out = np.zeros((4, 8)), np.zeros((4, 8))
    ref[:, [1, 2, 5]] = [.08, .05, .02]
    measured = score(dict(error=error(ref, out)), np.ones(4, dtype=bool))
    np.testing.assert_allclose(measured['R2'], .0003, atol=1e-15)
    policy = ActorCritic()
    with torch.no_grad():
        beta = policy.target(torch.zeros(2, 19))
    assert beta.shape == (2,) and bool(((beta > 0)&(beta < 1)).all())
    old = json.loads((HERE/'legacy_sources.json').read_text())
    assert sha(HERE/'legacy/repo/aav_mpc/condensed_mpc.py') == old['expected_core_sha256']['aav_mpc/condensed_mpc.py']
    for name in ('simulator/model.py', 'simulator/batch_mpc.py', 'simulator/rti_symbolic.py'):
        assert sha(ROOT/name) == old['expected_core_sha256'][name]
    print('PASS: imports, task IDs, reference derivatives, synthetic RMSE, beta dimensions and legacy source hashes.')
    print('No simulation, QP solve, dynamics rollout or training was executed.')
    print({name: len(tasks) for name, tasks in sets.items()})


if __name__ == '__main__':
    main()
