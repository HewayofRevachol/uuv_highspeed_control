"""Real on-policy UUV rollouts, local-teacher BC, and constrained PPO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.gs12_gate.data_io import file_hash, immutable_json, read_json
from .policy import SRPPOActorCritic
from .ppo import LagrangeState, srppo_update
from .storage import artifacts, load_run


def load_checkpoint(path, run):
    if path is None:
        raise ValueError('srppo evaluation requires --checkpoint')
    path = Path(path)
    load_run(run)
    metadata = read_json(path.with_suffix('.json'))
    if metadata['checkpoint_hash'] != file_hash(path):
        raise ValueError('checkpoint content hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['manifest_hash'] != file_hash(Path(run) / 'run_manifest.json'):
        raise ValueError('checkpoint/run manifest mismatch')
    policy = SRPPOActorCritic.from_checkpoint_config(saved['policy_config'])
    policy.load_state_dict(saved['policy']); policy.eval()
    return policy


def warm_start(policy, run, epochs, seed, learning_rate):
    """Fit reachable local teacher gates; no fake teacher-as-anchor identity."""
    root = Path(run) / 'teacher'
    summary = read_json(root / 'summary.json')
    if summary['source_split'] != 'train' or summary['manifest_hash'] != file_hash(Path(run) / 'run_manifest.json'):
        raise ValueError('teacher provenance mismatch')
    if summary['dataset_hash'] != file_hash(root / 'dataset.npz'):
        raise ValueError('teacher dataset hash mismatch')
    with np.load(root / 'dataset.npz') as data:
        obs = torch.as_tensor(data['observations'].copy())
        targets = torch.as_tensor(data['target_gate'].copy())
        risk_obs = torch.as_tensor(data['risk_observations'].copy())
        risk_targets = torch.as_tensor(data['risk_targets'].copy())
    if not len(obs) or epochs < 0:
        raise ValueError('invalid warm start dataset/epochs')
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    with torch.no_grad():
        before = float(policy.behavior_cloning_loss(obs, None, targets))
    rows = []
    for epoch in range(epochs):
        losses = []
        for idx in torch.randperm(len(obs)).split(256):
            loss = policy.behavior_cloning_loss(obs[idx], None, targets[idx])
            risk_idx = torch.randint(len(risk_obs), (len(idx),))
            risk_loss = torch.nn.functional.binary_cross_entropy(
                policy.values(risk_obs[risk_idx])['risk'], risk_targets[risk_idx])
            optimizer.zero_grad(set_to_none=True)
            (loss + .1 * risk_loss).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step(); losses.append(float(loss.detach()))
        rows.append({'epoch': epoch + 1, 'gate_loss': float(np.mean(losses))})
    with torch.no_grad():
        after = float(policy.behavior_cloning_loss(obs, None, targets))
    return {'epochs': epochs, 'samples': len(obs), 'loss_before': before, 'loss_after': after,
            'dataset_hash': summary['dataset_hash'], 'updates': rows,
            'risk_use': 'diagnostic_only; shield uses causal measured signals'}


def _batch(rows):
    keys = ('observations', 'teacher_gate', 'raw_z', 'old_logp', 'rewards', 'safety_cost',
            'value_track', 'value_safe', 'next_value_track', 'next_value_safe',
            'terminated', 'truncated')
    return {key: torch.as_tensor(np.asarray([row[key] for row in rows]),
                                dtype=torch.bool if key in ('terminated', 'truncated') else torch.float32)
            for key in keys}


def train(run, seed=41002, decisions=8192, update_every=512, teacher_warmup=None):
    from .episode import run_episode
    if decisions < 1 or update_every < 1 or decisions % update_every:
        raise ValueError('decisions must be a positive multiple of update_every')
    manifest, cases = load_run(run)
    cfg = manifest['config']
    train_cases = [c for c in cases if c['split'] == 'train' and not c['capacity_limited']]
    if not train_cases:
        raise ValueError('no feasible training cases')
    target = Path(run) / 'training' / str(seed)
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(1); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy = SRPPOActorCritic(observation_dim=cfg['observation_dim'], hidden=cfg['hidden'],
                             residual_limit=cfg['residual_limit'], cost_dim=cfg['cost_dim'])
    bc = warm_start(policy, run, cfg['teacher_warmup'] if teacher_warmup is None else teacher_warmup,
                    seed, cfg['learning_rate'])
    target.mkdir(parents=True)
    immutable_json(target / 'warm_start.json', bc)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg['learning_rate'])
    lagrange = LagrangeState.create(cfg['cost_dim'], cfg['safety_budget'],
                                   cfg['lagrange_lr'], cfg['lagrange_max'])
    episode_index = total_control = 0
    checkpoints = []
    started = time.perf_counter()
    # Shuffle whole manifest cycles, so every feasible stratum is encountered.
    order = rng.permutation(len(train_cases)); cursor = 0
    with (target / 'updates.jsonl').open('w', encoding='utf-8') as stream:
        for update in range(decisions // update_every):
            samples, episode_summaries = [], []
            while len(samples) < update_every:
                if cursor == len(order):
                    order = rng.permutation(len(train_cases)); cursor = 0
                case = train_cases[int(order[cursor])]; cursor += 1
                remaining = update_every - len(samples)
                folder = target / 'episodes' / f'{episode_index:06d}'
                summary = run_episode(case, 'srppo', folder, policy=policy, artifacts=artifacts(manifest),
                                      random_seed=int(rng.integers(2**31)),
                                      max_seconds=min(case['duration_s'], remaining * .5))
                with np.load(folder / 'decisions.npz') as data:
                    new = [{k: data[k][i] for k in data.files} for i in range(len(data['t']))]
                if not new or len(new) > remaining:
                    raise RuntimeError('episode violated on-policy batch size')
                samples.extend(new); episode_summaries.append(summary)
                episode_index += 1; total_control += summary['control_steps']
            metrics = srppo_update(policy, optimizer, _batch(samples), lagrange,
                                   epochs=cfg['policy_epochs'], minibatch=cfg['minibatch'],
                                   gamma=cfg['gamma'], lam=cfg['gae_lambda'], clip=cfg['clip'],
                                   value_weight=cfg['value_weight'], entropy_weight=cfg['entropy_weight'],
                                   grad_norm=cfg['grad_norm'])
            cumulative = (update + 1) * update_every
            entry = {'seed': seed, 'update': update + 1, 'decisions': cumulative,
                     'episodes': episode_index, 'control_steps': total_control,
                     'physical_failures': sum(s['first_failure_s'] is not None for s in episode_summaries),
                     'time_limit_truncations': sum(bool(s['truncated']) for s in episode_summaries),
                     'shield_override_fraction': float(np.mean([bool(r['override']) for r in samples])),
                     'wall_s': time.perf_counter() - started, **metrics}
            stream.write(json.dumps(entry, allow_nan=False) + '\n'); stream.flush()
            if update + 1 in {max(1, (decisions // update_every) // 2), decisions // update_every}:
                checkpoint = target / f'checkpoint_{cumulative:06d}.pt'
                payload = {'policy': policy.state_dict(), 'policy_config': policy.checkpoint_config(),
                           'optimizer': optimizer.state_dict(), 'lagrange': lagrange.state_dict(),
                           'torch_rng': torch.get_rng_state(), 'numpy_rng': rng.bit_generator.state,
                           'manifest_hash': file_hash(Path(run) / 'run_manifest.json'),
                           'teacher_hash': bc['dataset_hash'], 'seed': seed, 'decisions': cumulative}
                torch.save(payload, checkpoint)
                immutable_json(checkpoint.with_suffix('.json'), {'checkpoint_hash': file_hash(checkpoint),
                               'manifest_hash': payload['manifest_hash'], 'decisions': cumulative, 'seed': seed})
                checkpoints.append(str(checkpoint))
    result = {'mode': 'real_uuv_closed_loop', 'seed': seed, 'decisions': decisions,
              'updates': decisions // update_every, 'control_steps': total_control,
              'episodes': episode_index, 'checkpoints': checkpoints, 'warm_start': bc,
              'wall_s': time.perf_counter() - started}
    immutable_json(target / 'training_summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=41002)
    parser.add_argument('--decisions', type=int, default=8192)
    parser.add_argument('--update-every', type=int, default=512)
    parser.add_argument('--teacher-warmup', type=int)
    args = vars(parser.parse_args(argv))
    print(json.dumps(train(**args), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
