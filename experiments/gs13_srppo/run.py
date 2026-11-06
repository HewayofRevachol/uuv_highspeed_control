"""Prepare, collect real teacher labels and evaluate GS-13 closed-loop policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import torch

from experiments.gs12_gate.data_io import (file_hash, immutable_json, immutable_jsonl,
                                           read_json, write_npz)
from experiments.gs12_gate.features import schema
from .storage import PACKAGE, artifacts, load_run, source_hashes

BASELINES = {'hybrid_zero': 'hybrid_zero', 'fixed_best': 'fixed:1,1,1,0', 'rule_best': 'rule5'}
# All labels are reachable by the residual policy centred on fixed_best.
LOCAL_GATES = ((1., 1., 1., 0.), (.75, 1., 1., 0.), (1., .75, 1., 0.),
               (1., 1., .75, 0.), (1., 1., 1., .25), (.75, .75, .75, .25))


def prepare(out, config=PACKAGE / 'config.json', train_cases=None, dev_cases=None,
            test_cases=None, duration=None):
    from .sampler import build_cases
    out = Path(out)
    cfg = read_json(config)
    for key, value in [('train_cases', train_cases), ('dev_cases', dev_cases),
                       ('test_cases', test_cases), ('duration_s', duration)]:
        if value is not None:
            cfg[key] = value
    if out.exists():
        raise FileExistsError(f'new run directory required: {out}')
    if cfg['duration_s'] < .5 or abs(cfg['duration_s'] / .5 - round(cfg['duration_s'] / .5)) > 1e-8:
        raise ValueError('duration must be a positive multiple of 0.5 seconds')
    if any(cfg[f'{split}_cases'] < 1 for split in ('train', 'dev', 'test')):
        raise ValueError('every split must contain at least one case')
    cases = [c for split in ('train', 'dev', 'test') for c in
             build_cases(split, cfg[f'{split}_cases'], cfg['seed'], cfg['duration_s'])]
    out.mkdir(parents=True)
    feature_schema = schema()
    feature_schema['version'] = 'gs13_compatible_gs12_200_v1'
    manifest = {'protocol': cfg['version'], 'config': cfg,
                'cases_hash': immutable_jsonl(out / 'cases.jsonl', cases),
                'schema_hash': immutable_json(out / 'obs_schema.json', feature_schema),
                'source_hashes': source_hashes(),
                'normalization': cfg['normalization'],
                'cost_order': ['boundary', 'actuator', 'solver', 'fallback'],
                'baseline_provenance': 'GS12 fixed_best and rule5, no retuning on test',
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'torch': torch.__version__}}
    immutable_json(out / 'run_manifest.json', manifest)
    return {'run': str(out), 'cases': len(cases), 'manifest_hash': file_hash(out / 'run_manifest.json')}


def smoke(run):
    from .episode import run_episode
    from .policy import SRPPOActorCritic
    from experiments.gs12_gate.plant_harness import make_case
    manifest, _ = load_run(run)
    torch.manual_seed(manifest['config']['seed'])
    policy = SRPPOActorCritic()
    results = []
    for i, family in enumerate(('nominal', 'fin', 'tau')):
        case = make_case(f'smoke_{i}', 'smoke', 15., family, 'coupled', 91000 + i, duration_s=1.)
        for method in ('fixed:1,1,1,0', 'srppo_mean'):
            path = Path(run) / 'smoke' / case['case_id'] / method.replace(':', '_')
            summary = run_episode(case, method, path, policy=policy if method == 'srppo_mean' else None,
                                  artifacts=artifacts(manifest))
            with np.load(path / 'control.npz') as control:
                if not np.isfinite(control['issued']).all() or len(control['t']) != 10:
                    raise AssertionError('smoke control trace incomplete/nonfinite')
            with np.load(path / 'decisions.npz') as decisions:
                if not np.all((decisions['alpha'] >= 0) & (decisions['alpha'] <= 1)):
                    raise AssertionError('gate bounds failed')
            results.append({'case_id': case['case_id'], 'method': method,
                            'complete': summary['complete'], 'decisions': summary['decisions']})
    if not all(row['complete'] for row in results):
        raise AssertionError('smoke physical failure')
    immutable_json(Path(run) / 'smoke_summary.json', results)
    return results


def teacher_grid(run, limit=8):
    """Offline whole-episode local grid; no extra online MPC solves."""
    from .episode import run_episode
    manifest, cases = load_run(run)
    if limit < 1:
        raise ValueError('limit must be positive')
    selected = [c for c in cases if c['split'] == 'train' and not c['capacity_limited']][:limit]
    if not selected:
        raise ValueError('no feasible train cases')
    out = Path(run) / 'teacher'
    if (out / 'dataset.npz').exists():
        raise FileExistsError('teacher dataset exists; use a new run')
    observations, targets, risk_obs, risk_targets, labels = [], [], [], [], []
    for case in selected:
        candidates = []
        for j, gate in enumerate(LOCAL_GATES):
            folder = out / 'episodes' / case['case_id'] / str(j)
            method = 'fixed:' + ','.join(str(v) for v in gate)
            result = run_episode(case, method, folder, artifacts=artifacts(manifest))
            feasible = bool(result['complete'] and result['qp_failures'] == 0)
            candidates.append({'gate': gate, 'feasible': feasible,
                               'cost': result['tracking_cost_integral'],
                               'folder': str(folder.relative_to(Path(run)))})
            with np.load(folder / 'decisions.npz') as data:
                risk_obs.extend(data['observation'])
                risk_targets.extend([float(not feasible)] * len(data['t']))
        feasible_candidates = [row for row in candidates if row['feasible']]
        winner = min(feasible_candidates, key=lambda row: row['cost']) if feasible_candidates else None
        if winner is not None:
            with np.load(Path(run) / winner['folder'] / 'decisions.npz') as data:
                observations.extend(data['observation'])
                targets.extend([winner['gate']] * len(data['t']))
        labels.append({'case_id': case['case_id'], 'split': 'train', 'selected': winner,
                       'candidates': candidates, 'label_type': 'whole_episode_local_grid'})
    if not observations:
        raise RuntimeError('no safe teacher episode; inspect teacher/episodes before training')
    data_hash = write_npz(out / 'dataset.npz', observations=np.asarray(observations, np.float32),
                         target_gate=np.asarray(targets, np.float32),
                         risk_observations=np.asarray(risk_obs, np.float32),
                         risk_targets=np.asarray(risk_targets, np.float32))
    summary = {'source_split': 'train', 'cases': len(selected), 'labels': labels,
               'samples': len(observations), 'dataset_hash': data_hash,
               'manifest_hash': file_hash(Path(run) / 'run_manifest.json')}
    immutable_json(out / 'summary.json', summary)
    return {'cases': len(selected), 'samples': len(observations), 'dataset_hash': data_hash}


def evaluate(run, checkpoint, split='dev', methods='hybrid_zero,fixed_best,rule_best,srppo', limit=None):
    from .episode import run_episode
    from .train import load_checkpoint
    from .report import write_report
    manifest, cases = load_run(run)
    names = methods.split(',') if isinstance(methods, str) else methods
    if not names or set(names) - (set(BASELINES) | {'srppo'}):
        raise ValueError('unknown evaluation method')
    policy = load_checkpoint(checkpoint, run) if 'srppo' in names else None
    label = file_hash(checkpoint)[:12] if checkpoint else 'baselines'
    out = Path(run) / 'evaluation' / split / label
    if split == 'test' and 'srppo' in names:
        selection = read_json(Path(run) / 'selected_checkpoint.json')
        if selection['checkpoint_hash'] != file_hash(checkpoint):
            raise ValueError('test requires the checkpoint frozen by select on development results')
    selected = [c for c in cases if c['split'] == split]
    if limit is not None:
        if split == 'test':
            raise ValueError('test must use the whole frozen holdout')
        if limit < 1: raise ValueError('limit must be positive')
        selected = selected[:limit]
    records = []
    for case in selected:
        for name in names:
            method = 'srppo_mean' if name == 'srppo' else BASELINES[name]
            result = run_episode(case, method, out / 'episodes' / case['case_id'] / name,
                                 artifacts=artifacts(manifest), policy=policy if name == 'srppo' else None)
            failure = result['first_failure_s'] is not None
            duration = float(case['duration_s'])
            # Absorbing failure cost prevents short failed episodes looking cheap.
            penalty = (20. + 100. * max(0., duration - result['first_failure_s'])) if failure else 0.
            records.append({'scene_id': case['case_id'], 'method': name, 'split': split,
                            'capacity_limited': bool(case['capacity_limited']), 'failed': failure,
                            'truncated': result['truncated'], 'complete': result['complete'],
                            'cost': (result['tracking_cost_integral'] + penalty) / duration,
                            'tracking_integral': result['tracking_cost_integral'],
                            'qp_failures': result['qp_failures'], 'deadline_misses': result['deadline_misses'],
                            'saturation_fraction': result['saturation_fraction']})
    summary = write_report(records, out)
    immutable_json(out / 'provenance.json', {'split': split, 'full_split': limit is None,
                   'checkpoint': str(Path(checkpoint).resolve()) if checkpoint else None,
                   'checkpoint_hash': file_hash(checkpoint) if checkpoint else None,
                   'methods': names, 'manifest_hash': file_hash(Path(run) / 'run_manifest.json')})
    return summary


def select(run):
    """Freeze one checkpoint using complete development comparisons only."""
    load_run(run)
    candidates = []
    for folder in sorted((Path(run) / 'evaluation' / 'dev').glob('*')):
        provenance = read_json(folder / 'provenance.json')
        if not provenance['full_split'] or not provenance['checkpoint_hash']:
            continue
        report = read_json(folder / 'training_summary.json')
        score = report['strata']['feasible']['methods'].get('srppo')
        paired = report['strata']['feasible']['paired'].get('fixed_best', {}).get('srppo', {})
        if (score is not None and score.get('n', 0) > 0 and score.get('comparison_eligible', False)
                and paired.get('comparison_eligible', False)):
            candidates.append((score['failures'], score['mean_cost'], provenance))
    if not candidates:
        raise ValueError('run full dev evaluation with srppo before select')
    failures, cost, chosen = min(candidates, key=lambda x: (x[0], x[1], x[2]['checkpoint_hash']))
    selection = {**chosen, 'selection_split': 'dev', 'feasible_failures': failures, 'feasible_mean_cost': cost}
    immutable_json(Path(run) / 'selected_checkpoint.json', selection)
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=PACKAGE / 'config.json')
    for flag in ('train-cases', 'dev-cases', 'test-cases'):
        p.add_argument('--' + flag, type=int)
    p.add_argument('--duration', type=float)
    for command in ('smoke', 'teacher-grid', 'evaluate', 'select'):
        p = sub.add_parser(command); p.add_argument('--run', type=Path, required=True)
        if command == 'teacher-grid': p.add_argument('--limit', type=int, default=8)
        if command == 'evaluate':
            p.add_argument('--checkpoint', type=Path)
            p.add_argument('--split', choices=('dev', 'test'), default='dev')
            p.add_argument('--methods', default='hybrid_zero,fixed_best,rule_best,srppo')
            p.add_argument('--limit', type=int)
    args = vars(parser.parse_args(argv)); command = args.pop('command')
    print(json.dumps({'prepare': prepare, 'smoke': smoke, 'teacher-grid': teacher_grid,
                      'evaluate': evaluate, 'select': select}[command](**args), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
