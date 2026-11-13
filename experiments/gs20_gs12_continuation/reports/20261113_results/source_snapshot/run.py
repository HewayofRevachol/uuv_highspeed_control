"""Run explicitly requested cases, serially; no implicit experiments on import."""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import CONFIG, provenance, read_tasks, sha, write_json
from .references import frequency_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['first:0', 'first:1', 'hybrid:1'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--initial', choices=['common', 'endpoint'], default='common')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    from .closed_loop import Session
    tasks = read_tasks(args.tasks)[args.start:]
    if args.limit is not None:
        tasks = tasks[:args.limit]
    selection = json.loads(args.selection.read_text()) if args.selection else {}
    policy = None
    if args.policy:
        import torch
        from .policy import ActorCritic
        policy = ActorCritic()
        checkpoint = torch.load(args.policy, map_location='cpu', weights_only=False)
        policy.load_state_dict(checkpoint['agent'])
        policy.eval()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'traces').mkdir()
    write_json(args.out/'protocol.json', dict(tasks_sha256=sha(args.tasks), tasks=tasks,
               methods=args.methods, initial=args.initial, selection=selection,
               policy_sha256=sha(args.policy) if args.policy else None, provenance=provenance()))
    with (args.out/'episodes.jsonl').open('w') as stream:
        for task in tasks:
            for method in args.methods:
                structure, setting = method.split(':')
                beta = selection[structure]['fixed_beta'] if setting == 'best' else (
                       float(setting) if setting not in ('rule', 'rl') else CONFIG['initial_beta'])
                if policy is not None and setting == 'rl':
                    assert checkpoint['structure'] == structure
                initial = beta if args.initial == 'endpoint' else CONFIG['initial_beta']
                session = Session(task, structure, initial)
                while not session.done:
                    if setting == 'rule':
                        band = frequency_stage(task, session.k*.1+8.)
                        target = selection[structure]['rule_beta'][task['axis']][band]
                    elif setting == 'rl':
                        import torch
                        with torch.no_grad():
                            target = float(policy.target(torch.from_numpy(session.observe())[None])[0])
                    else:
                        target = beta
                    if args.initial == 'endpoint' and setting not in ('rule', 'rl'):
                        for _ in range(CONFIG['decision_cycles']):
                            if not session.done:
                                session.cycle(target, slew=False)
                    else:
                        session.step(target)
                record, arrays = session.result()
                name = f"{task['task_id']}__{method.replace(':', '_')}.npz"
                np.savez_compressed(args.out/'traces'/name, **arrays)
                record.update(method=method, trace=f'traces/{name}')
                stream.write(json.dumps(record)+'\n')
                stream.flush()
                print(json.dumps(dict(task_id=task['task_id'], method=method,
                                      complete=record['complete'], R2=record['R2'])), flush=True)


if __name__ == '__main__':
    main()
