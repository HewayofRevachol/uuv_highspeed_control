"""Causal SR-PPO rollouts on the existing GS-12 physical MPC/plant loop.

The actor sees the canonical measured 200-vector, never ``case['plant']``.
There is one MPC solve per valid 100 ms control period; the 500 ms gate
projection does not solve an additional candidate MPC problem.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from experiments.gs12_gate.data_io import digest, file_hash, immutable_json, read_json, write_npz
from experiments.gs12_gate.episode import (
    _error, _estimator_row, _make_observation,
    bootstrap_estimator_normalization, estimator_parameters,
    run_episode as run_baseline_episode,
)
from experiments.gs12_gate.features import CausalResidual, ObservationHistory, fast_reward
from experiments.gs12_gate.model_bridge import HybridMPC, THETA_SAFE, ThrustRateObserver, mix_gate
from experiments.gs12_gate.plant_harness import PlantHarness
from experiments.h15_mainline.response_estimator import SlidingWindowResponseEstimator
from experiments.gs13_srppo.costs import compute_cost, safety_vector
from experiments.gs13_srppo.safety import project_gate
from simulator.batch_mpc import BatchedUUVModel, CONTROL_SCALE
from simulator.model import rotation_ned

TEACHER_GATE = np.array([1., 1., 1., 0.])
BOUNDARY_LIMITS = np.array([8., 5., 2., np.deg2rad(40.)])


def boundary_errors(measurement):
    """Lateral, depth, tangent-speed and pitch, in the physical stop frame."""
    state, ref = measurement['state'], measurement['reference']
    if not np.isfinite(state).all() or not np.isfinite(ref).all():
        return np.full(4, np.nan)
    velocity = rotation_ned(state[3], state[4]) @ state[5:8]
    tangent = ref[5:8] / max(np.linalg.norm(ref[5:8]), 1e-12)
    lateral = -np.sin(ref[4]) * (state[0] - ref[0]) + np.cos(ref[4]) * (state[1] - ref[1])
    return np.array([lateral, state[2] - ref[2],
                     velocity @ tangent - np.linalg.norm(ref[5:8]), state[3]])


def failure_flags(measurement):
    errors = boundary_errors(measurement)
    return dict(zip(('lateral', 'depth', 'tangent_speed', 'pitch', 'nonfinite'),
                    [*(np.abs(errors) > BOUNDARY_LIMITS), not np.isfinite(errors).all()]))


def actuator_margins(actual):
    actual = np.asarray(actual, float)
    lower = np.array([0., -CONTROL_SCALE[1], -CONTROL_SCALE[2]])
    return np.minimum((actual - lower) / CONTROL_SCALE, (CONTROL_SCALE - actual) / CONTROL_SCALE)


def _observation(history, measurement, last_issued, innovation, theta, radius, alpha, qp_ok, qp_ms):
    try:
        observation = _make_observation(history, measurement, last_issued, innovation,
                                        theta, radius, alpha, qp_ok, qp_ms)[0]
        if observation.shape != (200,) or not np.isfinite(observation).all():
            raise ValueError('invalid observation')
        return observation, True
    except ValueError:
        # Nonfinite observations do not reach the actor. The transition is
        # explicitly marked invalid and emergency projection still receives
        # the original nonfinite marker.
        return np.zeros(200, dtype=np.float32), False


def _tensor(value, policy):
    parameter = next(policy.parameters())
    return torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)


def run_episode(case, method, out, *, artifacts=None, policy=None, random_seed=None,
                max_seconds=None, save_substeps=True):
    """Run SR-PPO or an unchanged GS-12 baseline and persist auditable arrays.

    ``srppo`` samples Gaussian residuals; ``srppo_mean`` is deterministic.
    An explicitly shortened collection interval is a bootstrap truncation.
    A full finite scenario horizon and any physical failure are terminals.
    """
    if method not in ('srppo', 'srppo_mean'):
        return run_baseline_episode(case, method, out, artifacts=artifacts, policy=policy,
                                    random_seed=random_seed, max_seconds=max_seconds,
                                    save_substeps=save_substeps)
    if policy is None:
        raise ValueError(f'{method} requires an SRPPO policy')
    # Local stochastic-rollout seeds do not reset the trainer's optimizer /
    # minibatch RNG stream. Policy probabilities remain those of raw_z.
    with torch.random.fork_rng(devices=[]):
        if random_seed is not None:
            torch.manual_seed(int(random_seed))
        return _run_srppo(case, method, out, artifacts or {}, policy, random_seed,
                          max_seconds, save_substeps)


def _run_srppo(case, method, out, artifacts, policy, random_seed, max_seconds, save_substeps):
    planned = float(case['duration_s'])
    limit = min(planned, float(max_seconds) if max_seconds is not None else planned)
    if not np.isfinite([planned, limit]).all() or limit <= 0.:
        raise ValueError('episode duration must be positive and finite')
    for value in (planned, limit):
        if abs(value / .1 - round(value / .1)) > 1e-8:
            raise ValueError('episode durations must be multiples of 100 ms')
    planned_steps = int(round(limit / .1))
    h = hashlib.sha256()
    for name, value in policy.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    policy_hash = h.hexdigest()
    signature = digest(dict(case=case, method=method, artifacts=artifacts,
                            policy_hash=policy_hash, random_seed=random_seed,
                            max_seconds=max_seconds, save_substeps=save_substeps,
                            adapter_version=1))
    out = Path(out)
    if out.exists():
        if (out / 'episode.json').exists():
            prior = read_json(out / 'episode.json')
            if prior.get('signature') == signature:
                for name, expected in prior['files'].items():
                    if not (out / name).exists() or file_hash(out / name) != expected:
                        raise ValueError(f'cached episode artifact hash mismatch: {out / name}')
                return prior
        raise FileExistsError(out)
    out.mkdir(parents=True)
    immutable_json(out / 'case.json', case)
    torch.set_num_threads(1)
    plant = PlantHarness(case)
    model = BatchedUUVModel('cpu')
    mpc = HybridMPC(model)
    trim_normalized = plant.trim / CONTROL_SCALE
    mpc.plan[:] = model.tensor(np.tile(trim_normalized, (35, 1)))
    mpc.previous[:] = model.tensor(trim_normalized)
    thrust_observer = ThrustRateObserver()
    thrust_observer.update(trim_normalized[0])
    residual = CausalResidual(mpc.dynamics, artifacts.get('residual_limits', [1., 1., 1., .1, .1]))
    history = ObservationHistory(artifacts.get('observation_scale'))
    estimator = SlidingWindowResponseEstimator(
        artifacts.get('estimator_normalization', bootstrap_estimator_normalization()),
        estimator_parameters(artifacts.get('bootstrap', True)))
    alpha, theta, radius = np.zeros(4), THETA_SAFE.copy(), np.array([.35, 2.])
    last_issued, last_qp_ok, last_qp_ms = plant.trim.copy(), True, 0.
    last_clip = False
    completed_row = None
    control, decisions, substeps = [], [], []
    first_failure, failure_reason = None, None
    started = time.perf_counter()

    for k in range(planned_steps):
        period_started = time.perf_counter()
        before = plant.measurement()
        estimator_started = time.perf_counter()
        if completed_row is not None:
            estimate = estimator.update(completed_row)
            theta = np.array([estimate.fin_gain, estimate.fin_tau_scale])
            radius = np.array([estimate.gain_radius, estimate.tau_radius])
            source, fallback_reason = estimate.source, estimate.fallback_reason or ''
        else:
            source, fallback_reason = 'E0', 'short_history'
        estimator_ms = 1000. * (time.perf_counter() - estimator_started)
        meaningful_fallback = source != 'E1' and fallback_reason not in ('', 'short_history')
        feature_started = time.perf_counter()
        innovation = residual.latest_ref()
        obs, obs_valid = _observation(history, before, last_issued, innovation,
                                      theta, radius, alpha, last_qp_ok, last_qp_ms)
        feature_ms = 1000. * (time.perf_counter() - feature_started)
        policy_ms = 0.
        control_override_reason = 'none'
        emergency = False
        previous_alpha = alpha.copy()
        if k % 5 == 0:
            raw_z, log_prob, value_track = np.zeros(4), 0., 0.
            value_safe, risk = np.zeros(4), 0.
            target = TEACHER_GATE.copy()
            policy_valid = obs_valid
            if obs_valid:
                policy_started = time.perf_counter()
                try:
                    with torch.no_grad():
                        result = policy.act(_tensor(obs[None], policy),
                                            _tensor(TEACHER_GATE, policy), deterministic=method == 'srppo_mean')
                    target = result['alpha_raw'][0].detach().cpu().numpy()
                    raw_z = result['raw_z'][0].detach().cpu().numpy()
                    log_prob = float(result['log_prob'][0])
                    value_track = float(result['value_track'][0])
                    value_safe = result['value_safe'][0].detach().cpu().numpy()
                    risk = float(result['risk'][0])
                    policy_valid = all(np.isfinite(value).all() for value in
                                       (target, raw_z, log_prob, value_track, value_safe, risk))
                except (ValueError, RuntimeError, FloatingPointError):
                    policy_valid = False
                if not policy_valid:
                    target = np.full(4, np.nan)
                    raw_z, log_prob, value_track, value_safe, risk = np.zeros(4), 0., 0., np.zeros(4), 1.
                policy_ms = 1000. * (time.perf_counter() - policy_started)
            measured_error = boundary_errors(before)
            margins = actuator_margins(before['actual'])
            projection = project_gate(alpha, target, dict(
                teacher_gate=TEACHER_GATE, observation=obs if obs_valid else [np.nan],
                qp_success=last_qp_ok, min_margin=float(np.min(margins)),
                boundary_ratio=float(np.max(np.abs(measured_error) / BOUNDARY_LIMITS)),
                risk=risk if artifacts.get('use_learned_risk', False) else 0.,
                saturation=bool(np.any(margins <= .02)),
                estimator_fallback=meaningful_fallback, residual_clip=last_clip))
            alpha = projection.alpha.copy()
            control_override_reason, emergency = projection.reason, projection.emergency
            valid = policy_valid and np.isfinite(raw_z).all() and np.isfinite(log_prob)
            decisions.append(dict(t=before['t'], observation=obs.copy(), observations=obs.copy(),
                                  raw_z=raw_z, log_prob=log_prob, old_logp=log_prob,
                                  value_track=value_track, value_safe=value_safe,
                                  target=target, teacher_gate=TEACHER_GATE.copy(),
                                  alpha=alpha.copy(), previous_alpha=previous_alpha,
                                  override=projection.overridden, override_reason=projection.reason,
                                  emergency_override=emergency, raw_valid=bool(valid), risk=risk,
                                  policy_version=policy_hash, reward=0., rewards=0.,
                                  reward_tracking=0., reward_alpha=-.01 * float(np.sum((alpha-previous_alpha)**2)),
                                  reward_failure=0., safety_cost=np.zeros(4), fast_steps=0))
            decisions[-1]['reward'] += decisions[-1]['reward_alpha']
        elif not last_qp_ok or not obs_valid:
            # Emergency changes can bypass the ordinary decision slew bound;
            # there is no additional policy sample and no additional solve.
            alpha = np.zeros(4)
            emergency = True
            control_override_reason = 'qp_failure' if not last_qp_ok else 'nonfinite_observation'
            decisions[-1]['override'] = True
            decisions[-1]['emergency_override'] = True
            decisions[-1]['override_reason'] += '+' + control_override_reason

        model_started = time.perf_counter()
        model_inputs_valid = np.isfinite(theta).all()
        theta_used = THETA_SAFE + alpha[0] * (theta - THETA_SAFE) if model_inputs_valid else THETA_SAFE.copy()
        d_ref, d_raw, d_clipped, was_clipped = residual.estimate(theta_used)
        model_inputs_valid = model_inputs_valid and np.isfinite(d_clipped).all()
        if model_inputs_valid:
            theta_used, d_used = mix_gate(alpha, theta, d_clipped)
        else:
            theta_used, d_used = THETA_SAFE.copy(), np.zeros(5)
            alpha = np.zeros(4)
            decisions[-1]['raw_valid'] = False
        model_alpha = alpha.copy()
        z = np.r_[before['state'], before['actual']/CONTROL_SCALE, thrust_observer.rate,
                  0., 0., before['queue'][:3].reshape(-1)/np.tile(CONTROL_SCALE, 3)]
        reference = plant.reference.horizon(before['t'])
        model_ms = 1000. * (time.perf_counter() - model_started)
        solve_started = time.perf_counter()
        solver_calls, qp_ok, qp_status = 0, False, 'nonfinite_input'
        if (model_inputs_valid and np.isfinite(z).all() and np.isfinite(reference).all()
                and np.isfinite(theta_used).all() and np.isfinite(d_used).all()):
            try:
                solver_calls = 1
                command, qp_flags = mpc.solve(model.tensor(z[None]), model.tensor(reference[None]),
                                             alpha=[1., 1.], theta_hat=theta_used,
                                             d_hat=np.tile(d_used, (35, 1)))
                issued = command[0].detach().cpu().numpy().copy()
                qp_ok = bool(qp_flags[0]) and np.isfinite(issued).all()
                qp_status = str(mpc.last_status[0])
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                qp_status = f'numerical_failure:{type(exc).__name__}'
        qp_ms = 1000. * (time.perf_counter() - solve_started)
        if not qp_ok:
            # The GS-12 solver already holds its last feasible command on QP
            # failure. Enforce the same finite hold for other numerical faults.
            issued = last_issued.copy()
            alpha = np.zeros(4)
            emergency, control_override_reason = True, 'qp_failure'
            decisions[-1]['override'] = True
            decisions[-1]['emergency_override'] = True
            decisions[-1]['override_reason'] += '+qp_failure'
        issued = np.clip(issued, [0., -CONTROL_SCALE[1], -CONTROL_SCALE[2]], CONTROL_SCALE)
        plan = mpc.plan[0].detach().cpu().numpy() * CONTROL_SCALE

        def stop_on_failure(sample):
            nonlocal first_failure, failure_reason
            flags = failure_flags(sample)
            if any(flags.values()):
                first_failure = sample['t']
                failure_reason = [name for name, hit in flags.items() if hit]
                return True
            return False

        plant_started = time.perf_counter()
        rows = plant.step(issued, stop_when=stop_on_failure)
        plant_ms = 1000. * (time.perf_counter() - plant_started)
        after = plant.measurement()
        if len(rows) == 5 and np.isfinite(after['state']).all():
            thrust_observer.update(after['actual'][0]/CONTROL_SCALE[0])
            residual.append(z, issued/CONTROL_SCALE, after['state'][5:10])
        errors_after, _ = _error(after)
        completed_row = (_estimator_row(before, after, issued, plan, errors_after)
                         if np.isfinite(after['state']).all() else None)
        saturation = float(np.any(np.isclose(issued, [0., -CONTROL_SCALE[1], -CONTROL_SCALE[2]], atol=1e-6) |
                                  np.isclose(issued, CONTROL_SCALE, atol=1e-6)))
        dt = len(rows) * .02
        if np.isfinite(errors_after).all() and np.isfinite(after['state'][3]):
            reward, stage_cost = fast_reward(errors_after, after['state'][3],
                                            (issued-last_issued)/CONTROL_SCALE, saturation, dt=dt)
        else:
            reward, stage_cost = -50. * dt, 50.
        physical_errors = boundary_errors(after)
        # Explicit catastrophic finite sentinels make the failure trainable
        # without putting NaN targets into either critic.
        cost = compute_cost(before, after, alpha, previous_alpha, issued,
                            dict(errors=np.nan_to_num(errors_after, nan=1e3, posinf=1e3, neginf=-1e3),
                                 boundary_errors=np.nan_to_num(physical_errors, nan=10., posinf=10., neginf=-10.),
                                 tracking_cost=stage_cost,
                                 actual=np.nan_to_num(after['actual']),
                                 margin=np.nan_to_num(actuator_margins(after['actual']), nan=-1.),
                                 qp_failure=not qp_ok, fallback=meaningful_fallback,
                                 residual_clip=was_clipped,
                                 deadline=(time.perf_counter()-period_started) > .1,
                                 failed=first_failure is not None, terminated=first_failure is not None,
                                 remaining_s=max(0., planned-after['t'])),
                            dict(actuator=CONTROL_SCALE, boundary=BOUNDARY_LIMITS))
        decisions[-1]['reward'] += reward - cost.terminal
        decisions[-1]['reward_tracking'] += reward
        decisions[-1]['reward_failure'] -= cost.terminal
        decisions[-1]['safety_cost'] += safety_vector(cost, dt)
        decisions[-1]['fast_steps'] += 1
        for sample in rows:
            if save_substeps:
                substeps.append(np.r_[sample['t'], sample['state'], sample['reference'],
                                      sample['actual'], sample['issued'],
                                      list(map(float, failure_flags(sample).values()))])
        control.append(dict(t=before['t'], state=before['state'], model_state=z,
                            actual=before['actual'], actual_after=after['actual'], queue=before['queue'],
                            issued=issued, mpc_command=plan[0], reference=before['reference'],
                            preview_0p5=before['preview_0p5'], preview_1p0=before['preview_1p0'],
                            errors=errors_after, boundary_errors=physical_errors,
                            theta_hat=theta.copy(), theta_used=theta_used, theta_radius=radius.copy(),
                            theta_source=source, estimator_fallback=fallback_reason, innovation=innovation,
                            d_ref=d_ref, d_raw=d_raw, d_used=d_used,
                            alpha=alpha.copy(), model_alpha=model_alpha,
                            qp_ok=qp_ok, qp_status=qp_status, solver_calls=solver_calls,
                            qp_primal_residual=float(mpc.last_primal_residual[0]),
                            qp_dual_residual=float(mpc.last_dual_residual[0]),
                            solve_ms=qp_ms, model_ms=model_ms, estimator_ms=estimator_ms,
                            feature_ms=feature_ms, policy_ms=policy_ms, plant_ms=plant_ms,
                            clip=was_clipped, saturation_fraction=saturation, reward=reward,
                            stage_cost=stage_cost, actual_dt=dt, residual_valid=len(rows)==5,
                            safety_cost=safety_vector(cost, dt), cost_components=cost.vector(),
                            shield_reason=control_override_reason, emergency_override=emergency,
                            total_ms=1000.*(time.perf_counter()-period_started),
                            deadline_miss=(time.perf_counter()-period_started) > .1))
        last_issued, last_qp_ok, last_qp_ms, last_clip = issued, qp_ok, qp_ms, was_clipped
        if first_failure is not None:
            break

    truncated = first_failure is None and limit < planned
    # The actual last measurement is retained for every episode, including
    # successful horizons. Only truncations consume its bootstrap values.
    if first_failure is None and completed_row is not None:
        estimate = estimator.update(completed_row)
        theta = np.array([estimate.fin_gain, estimate.fin_tau_scale])
        radius = np.array([estimate.gain_radius, estimate.tau_radius])
    final_obs, final_valid = _observation(history, plant.measurement(), last_issued,
                                        residual.latest_ref(), theta, radius, alpha, last_qp_ok, last_qp_ms)
    bootstrap_track, bootstrap_safe = 0., np.zeros(4)
    if truncated and final_valid:
        with torch.no_grad():
            values = policy.values(_tensor(final_obs[None], policy))
        bootstrap_track = float(values['value_track'][0])
        bootstrap_safe = values['value_safe'][0].detach().cpu().numpy()
    if truncated and not final_valid:
        raise ValueError('cannot bootstrap an invalid final observation')
    for j, decision in enumerate(decisions):
        last = j == len(decisions)-1
        decision['terminated'] = last and not truncated
        decision['terminal'] = decision['terminated']
        decision['truncated'] = last and truncated
        decision['rewards'] = decision['reward']
        decision['next_observation'] = final_obs if last else decisions[j+1]['observation']
        decision['next_value_track'] = bootstrap_track if last else decisions[j+1]['value_track']
        decision['next_value_safe'] = bootstrap_safe if last else decisions[j+1]['value_safe']
    write_npz(out/'control.npz', **{key: np.asarray([row[key] for row in control]) for key in control[0]})
    write_npz(out/'decisions.npz', **{key: np.asarray([row[key] for row in decisions]) for key in decisions[0]})
    files = ['case.json', 'control.npz', 'decisions.npz']
    if save_substeps:
        write_npz(out/'substeps.npz', values=np.asarray(substeps))
        immutable_json(out/'substep_schema.json', dict(columns=['t_s']+
                       [f'state_{i}' for i in range(10)]+[f'reference_{i}' for i in range(8)]+
                       [f'actual_{i}' for i in range(3)]+[f'issued_{i}' for i in range(3)]+
                       ['lateral_failure', 'depth_failure', 'tangent_speed_failure', 'pitch_failure', 'nonfinite_failure']))
        files += ['substeps.npz', 'substep_schema.json']
    summary = dict(case_id=case['case_id'], split=case['split'], method=method, signature=signature,
                   policy_hash=policy_hash, policy_variant='srppo', duration_expected_s=planned,
                   duration_actual_s=plant.t, complete=first_failure is None and not truncated,
                   truncated=truncated, first_failure_s=first_failure, failure_reason=failure_reason,
                   control_steps=len(control), substeps=len(substeps), decisions=len(decisions),
                   solver_calls=sum(row['solver_calls'] for row in control),
                   qp_failures=sum(not row['qp_ok'] for row in control),
                   deadline_misses=sum(row['deadline_miss'] for row in control),
                   saturation_fraction=float(np.mean([row['saturation_fraction'] for row in control])),
                   tracking_cost_integral=float(sum(row['actual_dt']*row['stage_cost'] for row in control)),
                   safety_cost_integral=np.sum([row['safety_cost'] for row in control], axis=0).tolist(),
                   shield_override_fraction=float(np.mean([row['override'] for row in decisions])),
                   emergency_overrides=sum(row['emergency_override'] for row in control),
                   wall_s=time.perf_counter()-started,
                   files={name: file_hash(out/name) for name in files})
    for column, name in ((0, 'peak_east_m'), (1, 'peak_depth_m')):
        errors = [abs(row['errors'][column]) for row in control if np.isfinite(row['errors'][column])]
        summary[name] = float(max(errors)) if errors else None
    immutable_json(out/'episode.json', summary)
    return summary
