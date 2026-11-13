"""Explicitly invoked S0 -> S1 -> S2 -> S3 pipeline; never runs on import."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .environment import BatchedEnv
from .features import OBSERVATION_VERSION
from .policy import ActorCritic, advantages, update
from .references import frozen_tasks, high_weight


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def checkpoint(path, agent, protocol, *, optimizer=None, policy_type='ppo', constant_beta=None):
    torch.save(dict(state_dict=agent.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
                    protocol=protocol, observation_version=OBSERVATION_VERSION,
                    policy_type=policy_type, constant_beta=constant_beta), path)


def load_policy(path, device='cuda'):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload['observation_version'] != OBSERVATION_VERSION:
        raise ValueError('checkpoint observation contract differs')
    agent = ActorCritic().to(device)
    agent.load_state_dict(payload['state_dict'])
    agent.eval()
    return agent, payload


def reward_curve(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (out/'rewards.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['episode', 'update', 'task_id', 'reward', 'J_track', 'failed'])
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    if rows:
        x = np.array([row['episode'] for row in rows])
        y = np.array([row['reward'] for row in rows])
        ax.plot(x, y, '.-', alpha=.45, label='episode return')
        ax.plot(x[7::8], [y[i:i+8].mean() for i in range(0, len(y), 8)], 'o-', label='8 episode mean')
        ax.legend()
    else:
        ax.text(.5, .5, 'S0 stopped: PPO not run', ha='center', va='center', transform=ax.transAxes)
    ax.set(xlabel='Training episode', ylabel='Integrated reward (higher is better)')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'reward_curve.png', dpi=160)
    plt.close(fig)


def rollout(tasks, method, out, options, *, agent=None, beta=.5, lead=0., training=False):
    env = BatchedEnv(tasks, **options)
    storage = {name: [] for name in ('observation', 'latent', 'logp', 'value', 'reward', 'done', 'valid')}
    with tqdm(total=env.cycles//10, desc=method+f' [{len(tasks)} lanes]', unit='s') as progress:
        while env.k < env.cycles:
            valid = env.alive.clone()
            observation = env.observe()
            with torch.no_grad():
                if agent is not None:
                    if training:
                        latent, target, logp, value = agent.sample(observation)
                    else:
                        target = agent.deterministic(observation)
                elif method.startswith('diagnostic'):
                    target = env.d.tensor([.5*float(high_weight(task, env.k*.1+lead)[0]) for task in tasks])
                else:
                    target = torch.full_like(env.beta, beta)
            learnable = training and env.k >= 80
            reward, done = env.step(target.double())
            if learnable:
                for name, value_ in [('observation', observation), ('latent', latent), ('logp', logp),
                                     ('value', value), ('reward', reward), ('done', done), ('valid', valid)]:
                    storage[name].append(value_)
            progress.update(1)
            if bool(done.all()):
                break
    summaries = env.summaries()
    write_json(out/f'{method}.json', summaries)
    np.savez_compressed(out/f'{method}_trace.npz', trace=env.trace[:env.k].cpu().numpy(),
                        columns=np.array(['time_s', 'beta', 'target', 'thrust', 'elevator', 'rudder',
                                          'error_E', 'error_D', 'error_VN', 'active', 'qp_primal', 'qp_dual']))
    if not training:
        return summaries
    if not storage['observation']:
        empty = torch.empty(0, device=env.d.device)
        return summaries, (empty.reshape(0, 78), empty, empty, empty, empty)
    data = {name: torch.stack(values) for name, values in storage.items()}
    advantage, returns = advantages(data['value'], data['reward'], data['done'], data['valid'])
    mask = data['valid'].flatten()
    batch = (data['observation'].reshape(-1, 78)[mask], data['latent'].flatten()[mask],
             data['logp'].flatten()[mask], returns.flatten()[mask], advantage.flatten()[mask])
    return summaries, batch


def ranking(rows):
    return (sum(row['failed'] for row in rows), sum(row['saturation_cycles'] > 0 for row in rows),
            sum(row['J_track'] for row in rows)/len(rows))


def opportunity_gate(fixed, diagnostic):
    reference_rank, probe_rank = ranking(fixed), ranking(diagnostic)
    wins = sum(b['J_track'] < a['J_track'] for a, b in zip(fixed, diagnostic))
    improvement = 1-probe_rank[2]/reference_rank[2] if reference_rank[2] else 0.
    passed = reference_rank[:2] == (0, 0) and probe_rank[:2] == (0, 0) and wins >= 3 and improvement >= .05
    return dict(passed=passed, paired_wins=wins, relative_improvement=improvement)


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(41001)
    np.random.seed(41001)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    options = dict(device=args.device, predictor=args.predictor, iterations=args.admm_iterations,
                   tolerance=args.admm_tolerance, cuda_graph=not args.no_cuda_graph)
    tasks = frozen_tasks()
    root = Path(__file__).resolve().parents[2]
    sources = list(Path(__file__).parent.glob('*.py'))+[root/'simulator'/name for name in ('model.py', 'batch_mpc.py', 'rti_symbolic.py')]
    source_hashes = {str(p.resolve().relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    protocol = dict(version='gs23-v1', observation_version=OBSERVATION_VERSION, tasks=tasks, seed=41001,
                    device=str(args.device), torch_version=str(torch.__version__),
                    gpu=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else None,
                    backend=options, source_hashes=source_hashes, horizon=35, control_dt=.1, plant_dt=.02,
                    decision_dt=1., warmup_s=8., initial_beta=.5, beta_slew=.02,
                    command_reward_scales=[1., 2., 2.], hard_command_slew=None,
                    state_weights=[3., 3., 3., 100., 30., 5., 5., 5.],
                    input_weights=[.15, .08, .08], rate_weights=[1.8, .35, .35], terminal_multiplier=2.5,
                    reward=dict(track_scale=1e6, beta_weight=.001, command_weight=.001,
                                cap=100., failure_extra=10., normalization='planned duration'),
                    ppo=dict(episodes=32, batch_episodes=8, epochs=4, minibatch=64, gamma=1., lam=.95,
                             learning_rate=.0003, clip=.2, initial_std=.15), trajectory_budget=100,
                    trace_columns='see each *_trace.npz columns', rmse_order=['N', 'E', 'D', 'pitch', 'yaw', 'VN', 'VE', 'VD'])
    write_json(out/'protocol.json', protocol)  # Freeze all splits, including holdout, before S0.
    agent = ActorCritic().to(args.device)
    checkpoint(out/'baseline_05.pt', agent, protocol, policy_type='constant', constant_beta=.5)
    fixed = {beta: rollout(tasks['dev'], f'fixed_{beta:g}', out, options, beta=beta) for beta in (0., .5, 1.)}
    diagnostic = {lead: rollout(tasks['dev'], f'diagnostic_{lead:g}', out, options, lead=lead) for lead in (0., 2.5)}
    best_beta = min(fixed, key=lambda beta: ranking(fixed[beta]))
    best_lead = min(diagnostic, key=lambda lead: ranking(diagnostic[lead]))
    gate = opportunity_gate(fixed[best_beta], diagnostic[best_lead])
    report = dict(stage='S0', trajectories=20, best_fixed_beta=best_beta, best_diagnostic_lead=best_lead, gate=gate)
    history = []
    reward_curve(out, history)
    write_json(out/'summary.json', report)
    if args.s0_only or not gate['passed']:
        report['stop_reason'] = 'S0 only requested' if args.s0_only else 'S0 opportunity threshold not met'
        checkpoint(out/'final_actor.pt', agent, protocol, policy_type='constant', constant_beta=best_beta)
        write_json(out/'summary.json', report)
        return report
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    generator = np.random.default_rng(41001)
    training_order = np.concatenate([generator.permutation(16) for _ in range(2)])
    updates = []
    for step in range(4):
        selected = [tasks['train'][int(i)] for i in training_order[step*8:step*8+8]]
        episodes, batch = rollout(selected, f'train_{step+1:02d}', out, options, agent=agent, training=True)
        # A completely failed warmup has no policy-controlled transitions.
        loss = update(agent, optimizer, batch) if len(batch[0]) else None
        for row in episodes:
            history.append({key: row[key] for key in ('task_id', 'reward', 'J_track', 'failed')} |
                           dict(episode=len(history)+1, update=step+1))
        updates.append(dict(update=step+1, loss=loss))
        checkpoint(out/f'checkpoint_{step+1:02d}.pt', agent, protocol, optimizer=optimizer)
        reward_curve(out, history)
        write_json(out/'updates.json', updates)
    report.update(stage='S2', trajectories=52)
    selection = []
    for step in range(4):
        policy, _ = load_policy(out/f'checkpoint_{step+1:02d}.pt', args.device)
        rows = rollout(tasks['dev'], f'dev_{step+1:02d}', out, options, agent=policy)
        selection.append(rows)
    best_step = min(range(4), key=lambda i: ranking(selection[i]))
    report.update(trajectories=68, selected_checkpoint=best_step+1)
    # Initial constant policy is a real selectable checkpoint using cached S0 results.
    baseline = min((.5, best_beta), key=lambda beta: ranking(fixed[beta]))
    improved = ranking(selection[best_step]) < ranking(fixed[baseline])
    report['rl_improved_dev'] = improved
    if not improved:
        report['stop_reason'] = 'PPO did not improve the fixed development baseline; holdout not opened'
        checkpoint(out/'final_actor.pt', agent, protocol, policy_type='constant', constant_beta=baseline)
        write_json(out/'summary.json', report)
        return report
    policy, _ = load_policy(out/f'checkpoint_{best_step+1:02d}.pt', args.device)
    checkpoint(out/'final_actor.pt', policy, protocol)
    holdout = dict(best_fixed=rollout(tasks['holdout'], 'holdout_fixed', out, options, beta=best_beta),
                   diagnostic=rollout(tasks['holdout'], 'diagnostic_holdout', out, options, lead=best_lead),
                   constant_05=rollout(tasks['holdout'], 'holdout_05', out, options, beta=.5),
                   ppo=rollout(tasks['holdout'], 'holdout_ppo', out, options, agent=policy))
    write_json(out/'holdout.json', holdout)
    report.update(stage='S3', trajectories=100, stop_reason='Frozen holdout completed; no further tuning')
    write_json(out/'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--predictor', choices=['cusadi', 'torch'], default='cusadi')
    parser.add_argument('--admm-iterations', type=int, default=400)
    parser.add_argument('--admm-tolerance', type=float, default=1e-5)
    parser.add_argument('--no-cuda-graph', action='store_true')
    parser.add_argument('--s0-only', action='store_true', help='run only the 20-trajectory opportunity check')
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
