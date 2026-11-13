"""Small solver/interface checks only: no PPO updates or complete trajectories."""
import numpy as np
import pytest
import torch

from simulator.rti_symbolic import actuator_aware_function
from .dynamics import Dynamics
from .environment import BatchedEnv, failure_penalty
from .features import Features
from .mpc import BoxADMM
from .policy import ActorCritic, advantages
from .references import frozen_tasks, reference
from .train import opportunity_gate, reward_curve


def test_projection_phase_and_alias():
    d = Dynamics('cpu')
    features = Features(d)
    omega = 2*np.pi/8
    now = 13.7
    t = torch.arange(80, dtype=torch.float64)*.1
    def signal(times):
        return (3+2*times+.006*torch.cos(omega*times+.3)+.002*torch.sin(3*omega*times-.4))
    future = signal(t+now)[None, :, None].expand(-1, -1, 3)
    past = signal(t+now-7.9)[None, :, None].expand(-1, -1, 3)
    torch.testing.assert_close(features.frequency(future), features.frequency(past, history=True), atol=1e-12, rtol=1e-9)
    lo, hi = torch.sin(omega*t), torch.sin(3*omega*t)
    assert torch.allclose(lo[[0, 40]], hi[[0, 40]], atol=1e-12)
    assert not torch.allclose(features.frequency(lo[None, :, None]), features.frequency(hi[None, :, None]))


def test_predictor_matches_existing_symbolic_model():
    d = Dynamics('cpu')
    x, actual, _, queue = d.initial([12., 18.])
    x[:, 3:] += d.tensor([.01, -.02, .1, .02, -.01, .003, -.002])
    z = torch.cat((x, actual/d.scale, (queue/d.scale).flatten(1)), -1)
    beta = d.tensor([0., 1.])
    command = d.tensor([[.3, .001, -.002], [.8, -.002, .001]])
    predicted = d.predictor_step(z, command, beta)
    function = actuator_aware_function(.1)
    for i in range(2):
        expected = np.array(function(z[i].numpy(), command[i].numpy(), [1-.15*float(beta[i]), 1+float(beta[i])], np.zeros(5))[0]).ravel()
        np.testing.assert_allclose(predicted[i], expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_admm_constrained_qp_against_osqp(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    import osqp
    import scipy.sparse as sp
    rng = np.random.default_rng(12)
    a = rng.normal(size=(2, 6, 6))
    h = np.swapaxes(a, -1, -2)@a+np.eye(6)
    g = rng.normal(size=(2, 6))*3
    lower, upper = -np.ones((2, 6))*.2, np.ones((2, 6))*.3
    tensors = [torch.tensor(v, device=device, dtype=torch.float64) for v in (h, g, lower, upper)]
    solver = BoxADMM(2, 6, iterations=200, tolerance=1e-7, cuda_graph=True)
    for multiplier in (1., .7):  # Verify replay after changing graph inputs.
        delta, valid, primal, dual = solver.solve(tensors[0], tensors[1]*multiplier, *tensors[2:])
        assert valid.all(), (primal, dual)
        for i in range(2):
            reference_solver = osqp.OSQP()
            reference_solver.setup(P=sp.csc_matrix(h[i]), q=g[i]*multiplier, A=sp.eye(6, format='csc'),
                                   l=lower[i], u=upper[i], eps_abs=1e-10, eps_rel=1e-10, verbose=False)
            expected = reference_solver.solve().x
            np.testing.assert_allclose(delta[i].cpu(), expected, atol=1e-7)


def test_reference_derivative_and_frozen_budget():
    tasks = frozen_tasks()
    assert [len(tasks[k]) for k in ('dev', 'train', 'holdout')] == [4, 16, 8]
    for task in tasks['dev']:
        t = np.array([3.2, 47.3, 49.1, 95., 97., 141.])
        numeric = (reference(task, t+1e-4)[:, :3]-reference(task, t-1e-4)[:, :3])/2e-4
        np.testing.assert_allclose(numeric, reference(task, t)[:, 5:], atol=1e-7)


def test_center_policy_gae_and_failure_cost():
    policy = ActorCritic()
    assert torch.equal(policy.deterministic(torch.randn(3, 78)), torch.full((3,), .5))
    assert torch.all(policy.target(torch.tensor([-100., 100.])) == torch.tensor([0., 1.]))
    values = torch.zeros(3, 2)
    rewards = torch.tensor([[1., -10.], [2., 0.], [3., 0.]])
    done = torch.tensor([[False, True], [False, True], [True, True]])
    valid = torch.tensor([[True, True], [True, False], [True, False]])
    advantage, _ = advantages(values, rewards, done, valid, lam=1.)
    torch.testing.assert_close(advantage, torch.tensor([[6., -10.], [5., 0.], [3., 0.]]))
    # Even the worst remaining clipped trajectory is preferable to failure.
    assert failure_penalty(40., 144.) < -100*(144-40)/144


def test_gate_and_plot_exports(tmp_path):
    fixed = [dict(J_track=1., failed=False, saturation_cycles=0) for _ in range(4)]
    probe = [dict(J_track=.9, failed=False, saturation_cycles=0) for _ in range(4)]
    assert opportunity_gate(fixed, probe)['passed']
    probe[0]['failed'] = True
    assert not opportunity_gate(fixed, probe)['passed']
    reward_curve(tmp_path, [dict(episode=i+1, update=1, task_id=str(i), reward=-1., J_track=1., failed=False) for i in range(8)])
    assert (tmp_path/'reward_curve.png').stat().st_size > 1000
    assert len((tmp_path/'rewards.csv').read_text().splitlines()) == 9


def test_qp_failure_is_charged_once_and_other_lane_continues():
    tasks = [frozen_tasks()['train'][0], frozen_tasks()['dev'][0]]
    env = BatchedEnv(tasks, device='cpu', predictor='torch', cuda_graph=False)
    def failed_lane(state, refs, beta, active):
        env.mpc.last_primal_residual = env.d.tensor([1., 0.])
        env.mpc.last_dual_residual = env.d.tensor([1., 0.])
        return env.last_command*env.d.scale, torch.tensor([False, True]) & active
    env.mpc.solve = failed_lane
    first = env.cycle(env.beta)
    assert float(first[0]) == -110.
    assert env.alive.tolist() == [False, True]
    second = env.cycle(env.beta)
    assert float(second[0]) == 0.
    assert env.samples.tolist() == [0., 10.]
    assert env.summaries()[0]['qp_failed']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_two_gpu_control_cycles_and_causal_observation():
    # Only 0.2 simulated seconds; no training or full experiment episode.
    tasks = [frozen_tasks()['train'][0], frozen_tasks()['dev'][0]]
    env = BatchedEnv(tasks, predictor='cusadi', iterations=400)
    observation = env.observe()
    assert observation.shape == (2, 78)
    assert torch.isfinite(observation).all()
    torch.testing.assert_close(observation[:, 32:34], torch.full((2, 2), .5, device='cuda'))
    # Changing hidden actuator rates must not change the policy observation.
    env.rate += 17.
    saved_queue = env.queue.clone()
    env.queue += 10.
    torch.testing.assert_close(env.observe(), observation)
    env.rate.zero_()
    env.queue.copy_(saved_queue)
    for _ in range(2):
        reward = env.cycle(torch.ones_like(env.beta))
        assert torch.isfinite(reward).all()
        assert env.qp_ok.all(), (env.mpc.last_primal_residual, env.mpc.last_dual_residual)
    torch.testing.assert_close(env.beta, env.d.tensor([.54, .54]))
    assert env.trace[:2].shape == (2, 2, 12)
    assert env.observe().shape == (2, 78)
    env.k = 80  # Exercise filled-history features without simulating more time.
    assert torch.isfinite(env.observe()).all()
