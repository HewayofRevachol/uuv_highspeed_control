"""On-policy recovery starts and short MPC-command faults for DDAT training."""

from collections import deque

import numpy as np
import torch


ENV_FIELDS = (
    "state", "parameters", "actuator", "command_history", "delay_steps",
    "tau", "disturbance", "previous_action", "base_command", "last_mpc_ok",
    "mpc_failure_streak", "path_ids", "progress", "steps", "max_steps",
    "finished", "disturbance_estimate", "current_ned", "current_estimate",
)
MPC_FIELDS = ("previous", "plan", "states", "last_primal_residual",
              "last_dual_residual", "last_qp_valid")
FSP_FIELDS = ("velocity_error", "expected_next", "expected_valid")


def snapshot(env):
    values = {name: getattr(env, name).detach().clone() for name in ENV_FIELDS}
    values["mpc"] = {name: getattr(env.mpc, name).detach().clone()
                     for name in MPC_FIELDS if hasattr(env.mpc, name)}
    if env.fsp is not None:
        values["fsp"] = {name: getattr(env.fsp, name).detach().clone()
                         for name in FSP_FIELDS}
    return values


def restore(env, indices, cases):
    """Restore independent environment rows; the RNG keeps generating fresh noise."""
    for index, case in zip(indices, cases):
        for name in ENV_FIELDS:
            getattr(env, name)[index] = case[name].to(env.device)
        for name, value in case["mpc"].items():
            getattr(env.mpc, name)[index] = value.to(env.device)
        if env.fsp is not None:
            if "fsp" in case:
                for name, value in case["fsp"].items():
                    getattr(env.fsp, name)[index] = value.to(env.device)
            else:
                mask = torch.zeros(env.batch_size, device=env.device, dtype=torch.bool)
                mask[index] = True
                env.fsp.reset(mask)


class RecoveryBuffer:
    def __init__(self, batch_size, lookback_min=5, lookback_max=20,
                 capacity=2048, seed=0):
        if not 1 <= lookback_min <= lookback_max or capacity < 1:
            raise ValueError("invalid recovery buffer limits")
        self.min_lag, self.max_lag = lookback_min, lookback_max
        self.history = deque(maxlen=lookback_max+1)
        self.cases = deque(maxlen=capacity)
        self.generation = torch.zeros(batch_size, dtype=torch.long)
        self.event_active = torch.zeros(batch_size, 4, dtype=torch.bool)
        self.rng = np.random.default_rng(seed)

    def record(self, env):
        self.history.append((self.generation.clone(), snapshot(env)))

    def collect(self, env, failed):
        """Save pre-event states, including states before solver failure starts."""
        _, _, path_pitch, _, _ = env.paths.sample(env.path_ids, env.progress[:, None])
        pitch_error = (env.state[:, 3]-path_pitch[:, 0]).abs()
        saturated = env.actuator[:, 1].abs() > .9*env.model.control_scale[1]
        signals = torch.stack((env.mpc_failure_streak >= 2,
                               pitch_error > np.deg2rad(10),
                               env.state[:, 8].abs() > .4, saturated), -1)
        event = (signals & ~self.event_active.to(env.device)).any(-1) | failed
        self.event_active = signals.cpu()
        for index in event.nonzero().flatten().tolist():
            current_step = int(env.steps[index])
            candidates = []
            for generation, state in self.history:
                lag = current_step-int(state["steps"][index])
                if (self.min_lag <= lag <= self.max_lag and
                    generation[index] == self.generation[index] and
                    not bool(state["finished"][index])):
                    candidates.append(state)
            if candidates:
                chosen = candidates[int(self.rng.integers(len(candidates)))]
                case = {name: chosen[name][index].cpu().clone() for name in ENV_FIELDS}
                case["mpc"] = {name: value[index].cpu().clone()
                               for name, value in chosen["mpc"].items()}
                if "fsp" in chosen:
                    case["fsp"] = {name: value[index].cpu().clone()
                                   for name, value in chosen["fsp"].items()}
                self.cases.append(case)

    def reset_rows(self, mask):
        mask = mask.cpu()
        self.generation[mask] += 1
        self.event_active[mask] = False

    def sample(self, count):
        indices = self.rng.integers(len(self.cases), size=count)
        return [self.cases[int(index)] for index in indices]

    def __len__(self):
        return len(self.cases)


class MPCFaultInjector:
    """Inject stale or biased commands immediately before a policy action."""

    def __init__(self, batch_size, device):
        self.remaining = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.age = torch.zeros_like(self.remaining)
        self.mode = torch.zeros_like(self.remaining)  # 1: stale; 2: elevator bias
        self.held_command = torch.zeros(batch_size, 3, device=device, dtype=torch.float64)
        self.elevator_bias = torch.zeros(batch_size, device=device, dtype=torch.float64)

    def clear(self, mask):
        self.remaining[mask] = 0
        self.age[mask] = 0
        self.mode[mask] = 0

    def schedule(self, env, indices, rng):
        for index in indices:
            self.mode[index] = int(rng.integers(1, 3))
            self.remaining[index] = int(rng.integers(3, 13))
            self.age[index] = 0
            self.held_command[index] = env.base_command[index]
            self.elevator_bias[index] = float(rng.uniform(-.35, .35))

    def apply(self, env):
        active = (self.remaining > 0) & ~env.finished
        stale = active & (self.mode == 1)
        biased = active & (self.mode == 2)
        if stale.any():
            env.base_command[stale] = self.held_command[stale]
            env.last_mpc_ok[stale] = False
            env.mpc_failure_streak[stale] = self.age[stale]+1
            if hasattr(env.mpc, "last_qp_valid"):
                env.mpc.last_qp_valid[stale] = True
                env.mpc.last_primal_residual[stale] = 10*env.mpc.tolerance
                env.mpc.last_dual_residual[stale] = 10*env.mpc.tolerance
        if biased.any():
            limit = env.model.control_scale[1]
            env.base_command[biased, 1] = (
                env.base_command[biased, 1]+self.elevator_bias[biased]
            ).clamp(-limit, limit)
        self.remaining[active] -= 1
        self.age[active] += 1


def recovery_reward(env):
    """Dense attitude and pitch-rate signal before the terminal pitch limit."""
    _, _, pitch, _, speed = env.paths.sample(env.path_ids, env.progress[:, None])
    future = env.progress[:, None]+speed*env.mpc.dt
    _, _, next_pitch, _, _ = env.paths.sample(env.path_ids, future)
    q_reference = (next_pitch[:, 0]-pitch[:, 0])/env.mpc.dt
    pitch_error = env.state[:, 3]-pitch[:, 0]
    rate_error = env.state[:, 8]-q_reference
    margin = (env.state[:, 3].abs()-np.deg2rad(40)).clamp_min(0)
    return (-.08*(pitch_error/.3).square() -
            .06*(rate_error/.3).square() -
            .2*(margin/np.deg2rad(20)).square()).float()
