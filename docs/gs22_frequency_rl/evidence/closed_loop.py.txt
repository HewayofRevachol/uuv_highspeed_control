"""Original GS-12 plant and private frozen QP; beta changes fin prediction only."""
import os
import time

import numpy as np
import torch

from .common import CONFIG, HERE, ROOT

os.environ['GS12_REPO'] = str(ROOT)

from .legacy.plant_harness import Plant, PlantConfig, diagnostic_values
from .legacy.repo.aav_mpc.condensed_mpc import CondensedActuatorMPC
from .legacy.hybrid_controller import HybridDynamics, ThrustRateObserver, make_hybrid_controller
from simulator.batch_mpc import BatchedUUVModel, CONTROL_SCALE
from .metrics import ERROR_SCALES, TRACK_SCALES, error, score, windows
from .references import reference

SCALE = np.asarray(CONTROL_SCALE)


class Session:
    def __init__(self, task, structure, initial_beta=1.):
        torch.set_num_threads(1)
        self.task, self.structure = task, structure
        self.model = BatchedUUVModel('cpu')
        if structure == 'first':
            self.mpc = CondensedActuatorMPC(self.model, 1, horizon=35, dt=.1)
            self.observer = None
        else:
            self.mpc = make_hybrid_controller(self.model, HERE/'legacy/repo')
            # Unit base parameters are required for the 0 -> 1 beta interpolation.
            self.mpc.dynamics = HybridDynamics(fin_gain_scale=1., fin_tau_scale=1.)
            self.observer = ThrustRateObserver()
        self.plant = Plant(task['speed_mps'], PlantConfig(**CONFIG['plants'][task['plant']]))
        self.mpc.previous[:] = self.model.tensor(self.plant.actual/SCALE)
        self.mpc.plan[:] = self.mpc.previous[:, None]
        self.initial_control = (self.plant.actual/SCALE).copy()
        self.estimated_rate = np.zeros(3)
        self.beta, self.previous_target = initial_beta, initial_beta
        self.k = 0
        self.violation = None
        self.started = time.perf_counter()
        self.fields = {key: [] for key in ('time', 'state', 'output', 'reference', 'error',
            'diagnostic', 'actual', 'command', 'qp_ok', 'beta', 'beta_target',
            'qp_primal', 'qp_dual', 'estimated_rate')}

    @property
    def done(self):
        return self.k >= round(self.task['duration_s']/.1) or self.violation is not None

    def observe(self):
        t = self.k*.1
        ref = reference(self.task, t)
        e = error(ref, self.plant.output())/ERROR_SCALES
        preview = reference(self.task, t+np.array([4., 8.]))
        changes = (preview[:, [1, 2, 5]]-ref[[1, 2, 5]])/TRACK_SCALES
        return np.r_[e, self.beta, self.previous_target, self.plant.actual/SCALE,
                     changes.ravel()].astype(np.float32)

    def cycle(self, target, slew=True):
        self.beta = self.beta + np.clip(target-self.beta, -.02, .02) if slew else target
        z = self.plant.controller_state()
        if self.observer is not None:
            z = np.r_[z, self.estimated_rate]
        refs = reference(self.task, (self.k+np.arange(1, 36))*.1)
        command, ok = self.mpc.solve(self.model.tensor(z)[None], self.model.tensor(refs)[None],
                                    alpha=[self.beta, 0.], theta_hat=[.85, 2.])
        command = command[0].numpy()
        old_actual = (self.plant.actual/SCALE).copy()
        old_queue = (self.plant.queue[:3]/SCALE).copy()
        f = self.fields
        for name, value in [('command', command), ('qp_ok', bool(ok[0])), ('beta', self.beta),
                ('beta_target', target), ('qp_primal', float(self.mpc.last_primal_residual[0])),
                ('qp_dual', float(self.mpc.last_dual_residual[0]))]:
            f[name].append(value)
        for j in range(5):
            self.plant.step(command)
            t = (5*self.k+j+1)*.02
            ref, out = reference(self.task, t), self.plant.output()
            diag = diagnostic_values(out, ref)
            for name, value in [('time', t), ('state', self.plant.x.copy()), ('output', out),
                    ('reference', ref), ('error', error(ref, out)), ('diagnostic', diag),
                    ('actual', self.plant.actual.copy()), ('estimated_rate', self.estimated_rate.copy())]:
                f[name].append(value)
            exceeded = np.abs(diag) > [8., 5., 2., np.deg2rad(40)]
            if np.any(exceeded) or not np.isfinite(self.plant.x).all():
                self.violation = dict(time_s=t, diagnostic=diag.tolist())
                break
        if self.observer is not None and self.violation is None:
            self.estimated_rate = self.observer.estimate(old_actual, self.plant.actual/SCALE,
                                                        old_queue, command/SCALE)
        self.k += 1
        self.previous_target = target

    def step(self, target):
        start = len(self.fields['error'])
        for _ in range(CONFIG['decision_cycles']):
            if self.done:
                break
            self.cycle(float(target))
        e = np.asarray(self.fields['error'][start:])[:, [1, 2, 5]]/TRACK_SCALES
        reward = -CONFIG['reward_scale']*float(np.sum(e**2))/(self.task['duration_s']/.02)
        if self.violation is not None or (self.done and not all(self.fields['qp_ok'])):
            reward -= CONFIG['failure_penalty']
        return reward

    def result(self):
        arrays = {k: np.asarray(v) for k, v in self.fields.items()}
        saturation = lambda v: np.mean((v >= 1-1e-6)|(v <= np.array([0., -1., -1.])+1e-6), axis=0).tolist()
        complete = self.violation is None and self.k == round(self.task['duration_s']/.1)
        failed = int(np.sum(~arrays['qp_ok']))
        scores = {name: score(arrays, mask) for name, mask in windows(self.task, arrays['time']).items()}
        record = dict(task=self.task, structure=self.structure, complete=complete,
                      first_violation=self.violation, qp_failures=failed,
                      observed_end_s=float(arrays['time'][-1]),
                      R2=scores['full']['R2'] if complete and failed == 0 else None,
                      scores=scores, command_saturation=saturation(arrays['command']/SCALE),
                      actual_saturation=saturation(arrays['actual']/SCALE),
                      max_command_slew=np.max(np.abs(np.diff(
                          np.vstack((self.initial_control, arrays['command']/SCALE)),
                          axis=0)), axis=0).tolist(),
                      wall_s=time.perf_counter()-self.started)
        return record, arrays
