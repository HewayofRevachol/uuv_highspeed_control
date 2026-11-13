"""Aggregate physical RMSE, shared-prefix diagnostics and frozen fixed-beta choices."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import CONFIG, sha, write_json
from .metrics import OUTPUT_NAMES, score


def load_rows(directories):
    rows = []
    for directory in directories:
        for line in (directory/'episodes.jsonl').read_text().splitlines():
            row = json.loads(line)
            row['_directory'] = directory
            rows.append(row)
    keys = [(r['task']['task_id'], r['method']) for r in rows]
    assert len(keys) == len(set(keys)), 'Do not mix duplicate task/method runs.'
    return rows


def eligible(row):
    return row['R2'] is not None and not any(row['command_saturation']+row['actual_saturation'])


def freeze(rows, path, sources, rules_from=None):
    assert {r['task']['split'] for r in rows} == {'dev'}, 'Freeze from development tasks only.'
    result = {}
    task_ids = {r['task']['task_id'] for r in rows}
    for structure in ('first', 'hybrid'):
        candidates = {}
        for beta in CONFIG['betas']:
            method = f'{structure}:{beta:g}'
            selected = [r for r in rows if r['method'] == method]
            if {r['task']['task_id'] for r in selected} == task_ids and all(eligible(r) for r in selected):
                candidates[beta] = float(np.mean([r['R2'] for r in selected]))
        best = min(candidates, key=candidates.get)
        rule = json.loads(rules_from.read_text())[structure]['rule_beta'] if rules_from else {}
        if rules_from is None:
            for axis in ('east', 'depth', 'coupled'):
                rule[axis] = {}
                for k in (8, 24):
                    by_beta = {}
                    for beta in candidates:
                        selected = [r for r in rows if r['method'] == f'{structure}:{beta:g}'
                                    and r['task']['kind'] == 'single' and r['task']['axis'] == axis
                                    and r['task']['k'] == k]
                        by_beta[beta] = float(np.mean([r['R2'] for r in selected]))
                    rule[axis][str(k)] = min(by_beta, key=by_beta.get)
        result[structure] = dict(fixed_beta=best, dev_mean_R2=candidates[best],
                                 fixed_grid_mean_R2=candidates, rule_beta=rule)
    result['sources'] = {str(p/'episodes.jsonl'): sha(p/'episodes.jsonl') for p in sources}
    result['rules_sha256'] = sha(rules_from) if rules_from else None
    result['selection_metric'] = 'task-equal mean full-episode R2; all tasks complete, no QP failure or saturation'
    write_json(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--rules-from', type=Path, help='Keep the rule from a previous static development selection.')
    parser.add_argument('--selection', type=Path, help='Frozen baseline when these runs contain a numeric beta grid.')
    args = parser.parse_args()
    rows = load_rows(args.runs)
    args.out.mkdir(parents=True, exist_ok=False)
    common_end = {tid: min(r['observed_end_s'] for r in rows if r['task']['task_id'] == tid)
                  for tid in {r['task']['task_id'] for r in rows}}
    flat = []
    for r in rows:
        arrays = np.load(r['_directory']/r['trace'])
        shared = score(arrays, arrays['time'] <= common_end[r['task']['task_id']]+1e-9)
        full = r['scores']['full']
        flat.append(dict(task_id=r['task']['task_id'], method=r['method'],
            kind=r['task']['kind'], axis=r['task'].get('axis', r['task'].get('maneuver')),
            k=r['task'].get('k'), speed_mps=r['task']['speed_mps'], plant=r['task']['plant'],
            amplitude=r['task']['amplitude'], phase_rad=r['task'].get('phase_rad'),
            depth_phase_offset=r['task'].get('depth_phase_offset', 0.), split=r['task']['split'],
            complete=r['complete'], eligible=eligible(r), qp_failures=r['qp_failures'],
            observed_end_s=r['observed_end_s'], R2=r['R2'],
            R=full['R'] if r['R2'] is not None else None,
            common_end_s=common_end[r['task']['task_id']], common_R=shared['R'],
            **{name: full['rmse'][name] if r['R2'] is not None else None for name in OUTPUT_NAMES},
            **{f'common_{name}': shared['rmse'][name] for name in OUTPUT_NAMES}))
    with (args.out/'rmse.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    summary = {}
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method]
        complete = [r for r in selected if r['R2'] is not None]
        summary[method] = dict(tasks=len(selected), complete=len(complete),
            eligible=sum(eligible(r) for r in selected), qp_failures=sum(r['qp_failures'] for r in selected),
            mean_R2_complete_only=float(np.mean([r['R2'] for r in complete])) if complete else None)
    if args.freeze:
        freeze(rows, args.freeze, args.runs, args.rules_from)
    selection_path = args.freeze or args.selection
    selection = json.loads(selection_path.read_text()) if selection_path else {}
    # Every dynamic/best/RL method is compared with the frozen baseline on identical tasks.
    paired = {}
    for structure in ('first', 'hybrid'):
        baseline = {r['task']['task_id']: r for r in rows if r['method'] == f'{structure}:best'}
        if not baseline and structure in selection:
            method = f"{structure}:{selection[structure]['fixed_beta']:g}"
            baseline = {r['task']['task_id']: r for r in rows if r['method'] == method}
        for setting in ('rule', 'rl'):
            candidate = {r['task']['task_id']: r for r in rows if r['method'] == f'{structure}:{setting}'}
            ids = sorted(baseline.keys() | candidate.keys())
            valid = [tid for tid in ids if tid in baseline and tid in candidate
                     and eligible(baseline[tid]) and eligible(candidate[tid])]
            if valid:
                b = np.array([baseline[t]['R2'] for t in valid])
                c = np.array([candidate[t]['R2'] for t in valid])
                paired[f'{structure}:{setting}'] = dict(pairs=len(ids), eligible_pairs=len(valid),
                    all_pairs_eligible=len(valid) == len(ids), wins=int(np.sum(c < b)),
                    mean_R2_reduction_fraction=float(1-c.mean()/b.mean()),
                    per_task=[dict(task_id=t, baseline_R2=float(x), candidate_R2=float(y))
                              for t, x, y in zip(valid, b, c)])
    write_json(args.out/'summary.json', dict(methods=summary, paired=paired,
        note='Partial/failed runs are excluded from full scores; common-prefix RMSE is diagnostic only.'))
    print(summary)


if __name__ == '__main__':
    main()
