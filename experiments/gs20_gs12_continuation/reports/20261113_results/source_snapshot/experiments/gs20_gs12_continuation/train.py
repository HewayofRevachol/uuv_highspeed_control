"""CPU PPO over the same plant and frozen QP. Explicit invocation runs training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import CONFIG, provenance, read_tasks, sha, write_json
from .policy import ActorCritic, update


def rollout(agent, task, structure):
    from .closed_loop import Session
    session = Session(task, structure, initial_beta=CONFIG['initial_beta'])
    steps = []
    while not session.done:
        observation = session.observe()
        with torch.no_grad():
            obs = torch.from_numpy(observation)[None]
            dist = agent.distribution(obs)
            raw = dist.sample()
            value = float(agent.value(obs)[0])
            logp = float(dist.log_prob(raw)[0])
        reward = session.step(float(raw.sigmoid()[0]))
        steps.append((observation, float(raw[0]), logp, value, reward))
    # gamma=1: the undiscounted tracking reward sums to -scale * full R2.
    rows, gae, next_value = [], 0., 0.
    for observation, raw, logp, value, reward in reversed(steps):
        gae = reward+next_value-value+.95*gae
        rows.append((observation, raw, logp, gae+value, gae))
        next_value = value
    record, _ = session.result()
    return list(reversed(rows)), dict(task_id=task['task_id'], complete=record['complete'],
                                     R2=record['R2'], qp_failures=record['qp_failures'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--structure', choices=['first', 'hybrid'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=41001)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    tasks = read_tasks(args.tasks)
    assert {t['split'] for t in tasks} == {'train'}
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    agent = ActorCritic()
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'protocol.json', dict(structure=args.structure, seed=args.seed,
        episodes=args.episodes, batch=args.batch, tasks_sha256=sha(args.tasks),
        reward='-1e6 * sum(normalized three-output squared errors) / requested substeps; failure -100',
        gamma=1., gae_lambda=.95, epochs=4, learning_rate=3e-4, provenance=provenance()))
    history = []
    with (args.out/'episodes.jsonl').open('w') as stream:
        for start in range(0, args.episodes, args.batch):
            batch = []
            for _ in range(min(args.batch, args.episodes-start)):
                task = tasks[int(rng.integers(len(tasks)))]
                transitions, record = rollout(agent, task, args.structure)
                batch.extend(transitions)
                stream.write(json.dumps(record)+'\n')
                stream.flush()
                print(json.dumps(record), flush=True)
            loss = update(agent, optimizer, batch)
            history.append(dict(episodes=min(start+args.batch, args.episodes), loss=loss))
            torch.save(dict(agent=agent.state_dict(), seed=args.seed, structure=args.structure,
                            episodes=history[-1]['episodes']), args.out/'latest.pt')
            write_json(args.out/'history.json', history)
    torch.save(dict(agent=agent.state_dict(), seed=args.seed, structure=args.structure,
                    episodes=args.episodes), args.out/'final.pt')


if __name__ == '__main__':
    main()
