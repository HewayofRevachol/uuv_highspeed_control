"""Reproducible GS-12 research plant, independent of production sources.

All factors are relative to simulator.model.MISMATCHED_PLANT. The controller
always receives its own fixed three-slot command history, including when the
true delay is different. RK4 and actuator recurrences match the source code.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1] / "uuv_highspeed_control"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from simulator.batch_mpc import BatchedUUVModel, CONTROL_SCALE
from simulator.model import D_COEFF, ELEVATOR_COEFF, MASS, M_EFF, RUDDER_COEFF
from simulator.model import MISMATCHED_PLANT as BASE
from simulator.rti_symbolic import ActuatorAwareCpuDynamics
from aav_mpc.condensed_mpc import CondensedActuatorMPC

SCALE = np.asarray(CONTROL_SCALE)
DT = .02
PERIOD = .1
OUTPUTS = ["north_m", "east_m", "depth_m", "pitch_rad", "yaw_rad",
           "VN_mps", "VE_mps", "VD_mps"]


@dataclass(frozen=True)
class PlantConfig:
    mass: float = 1.
    damping: float = 1.
    thrust: float = 1.
    elevator: float = 1.
    rudder: float = 1.
    delay_steps: tuple = (3, 3, 3)
    tau_scale: tuple = (1., 1., 1.)
    order: str = "first"

    def __post_init__(self):
        if self.order not in ("first", "second_all", "second_fins"):
            raise ValueError("unsupported actuator order")
        gains = [self.mass, self.damping, self.thrust, self.elevator, self.rudder]
        if not np.isfinite(gains).all() or min(gains) <= 0:
            raise ValueError("multipliers must be positive and finite")
        if len(self.delay_steps) != 3 or any(int(d) != d or d < 0 for d in self.delay_steps):
            raise ValueError("three nonnegative integer delay steps required")
        if len(self.tau_scale) != 3 or min(self.tau_scale) <= 0:
            raise ValueError("three positive tau scales required")

    @property
    def parameters(self):
        return np.asarray([BASE.mass_scale*self.mass, BASE.damping_scale*self.damping,
                           BASE.thrust_gain*self.thrust, BASE.elevator_gain*self.elevator,
                           BASE.rudder_gain*self.rudder])


@lru_cache(maxsize=1)
def hull_functions():
    import casadi as ca
    x = ca.SX.sym("x", 10)
    control = ca.SX.sym("physical_control", 3)
    parameters = ca.SX.sym("parameters", 5)
    inverse_mass = ca.DM(np.linalg.inv(M_EFF))

    def rhs(state):
        theta, psi = state[3], state[4]
        u, v, w, q, r = (state[i] for i in range(5, 10))
        ct, st, cp, sp = ca.cos(theta), ca.sin(theta), ca.cos(psi), ca.sin(psi)
        velocity = ca.vertcat(cp*ct*u-sp*v+cp*st*w,
                             sp*ct*u+cp*v+sp*st*w, -st*u+ct*w)
        coriolis = parameters[0]*MASS*ca.vertcat(-r*v+q*w, r*u, -q*u, 0, 0)
        damping = parameters[1]*u*ca.DM(D_COEFF)@state[5:10]
        force = ca.vertcat(parameters[2]*control[0], 0, 0, 0, 0)
        force += u*u*(parameters[3]*ca.DM(ELEVATOR_COEFF)*control[1]
                      + parameters[4]*ca.DM(RUDDER_COEFF)*control[2])
        acceleration = (inverse_mass@(force-coriolis-damping))/parameters[0]
        return ca.vertcat(velocity, q, r/ct, acceleration)

    k1 = rhs(x)
    k2 = rhs(x+DT*k1/2)
    k3 = rhs(x+DT*k2/2)
    k4 = rhs(x+DT*k3)
    next_x = x+DT*(k1+2*k2+2*k3+k4)/6
    return (ca.Function("gs12_hull_step", [x, control, parameters], [next_x]),
            ca.Function("gs12_hull_rhs", [x, control, parameters], [rhs(x)]))


class Plant:
    def __init__(self, speed=15., config=PlantConfig()):
        self.config = config
        self.parameters = config.parameters
        # Deliberately identical baseline trim across all true plant factors.
        trim = BASE.damping_scale*D_COEFF[0, 0]*speed**2/BASE.thrust_gain
        self.x = np.array([0., 0., 20., 0., 0., speed, 0., 0., 0., 0.])
        self.actual = np.array([trim, 0., 0.])
        self.rate = np.zeros(3)
        self.queue = np.tile(self.actual, (max(3, max(config.delay_steps)), 1))
        self.tau = np.array([BASE.thrust_tau_s, BASE.fin_tau_s, BASE.fin_tau_s])*config.tau_scale
        self.step_function, self.rhs_function = hull_functions()

    def controller_state(self):
        return np.r_[self.x, self.actual/SCALE, (self.queue[:3]/SCALE).ravel()]

    def output(self):
        derivative = np.asarray(self.rhs_function(self.x, self.actual, self.parameters)).ravel()
        return np.r_[self.x[:5], derivative[:3]]

    def step(self, command):
        command = np.asarray(command)
        target = np.array([command[j] if delay == 0 else self.queue[delay-1, j]
                           for j, delay in enumerate(self.config.delay_steps)])
        self.queue[1:] = self.queue[:-1].copy()
        self.queue[0] = command
        first = self.actual + (-np.expm1(-DT/self.tau))*(target-self.actual)
        if self.config.order == "first":
            self.actual = first
        else:
            omega = 1./np.maximum(self.tau, .03)
            self.rate += DT*(omega**2*(target-self.actual)-2.*omega*self.rate)
            second = self.actual+DT*self.rate
            self.actual = second if self.config.order == "second_all" else np.r_[first[0], second[1:]]
            if self.config.order == "second_fins":
                self.rate[0] = 0.
        self.x = np.asarray(self.step_function(self.x, self.actual, self.parameters)).ravel()
        return self.x.copy()


def reference(t, speed=15., maneuver="coupled"):
    """Analytic, exogenous reference. Velocity is exactly position derivative."""
    t = np.asarray(t, dtype=float)
    zero = np.zeros_like(t)
    north, vn = speed*t, zero+speed
    if maneuver == "straight":
        north = north + 1.2*(1-np.cos(.5*t))
        vn = vn + .6*np.sin(.5*t)
    elif maneuver not in ("horizontal", "depth", "coupled"):
        raise ValueError("unknown maneuver")
    east = 6.*(1-np.cos(.7*t)) if maneuver in ("horizontal", "coupled") else zero
    ve = 4.2*np.sin(.7*t) if maneuver in ("horizontal", "coupled") else zero
    depth = 20 + (2.4*(1-np.cos(.84*t)) if maneuver in ("depth", "coupled") else zero)
    vd = 2.016*np.sin(.84*t) if maneuver in ("depth", "coupled") else zero
    pitch = -np.arctan2(vd, np.hypot(vn, ve))
    yaw = np.arctan2(ve, vn)
    return np.stack((north, east, depth, pitch, yaw, vn, ve, vd), axis=-1)


def diagnostic_values(output, ref):
    position_error = output[:3]-ref[:3]
    side = np.array([-np.sin(ref[4]), np.cos(ref[4]), 0.])
    tangent = ref[5:]/np.linalg.norm(ref[5:])
    return np.array([side@position_error, position_error[2],
                     tangent@output[5:]-np.linalg.norm(ref[5:]), output[3]])


def run_closed_loop(config=PlantConfig(), speed=15., reference=None, duration=24.,
                    model_gain=1., model_tau=1., mpc_setup=None,
                    stop_at_limit=True):
    """Run original MPC. reference is callable(t)->[...,8]; setup(mpc) may tune it.

    Returns {meta, arrays}; arrays contain every 0.02 s substep and every 0.1 s
    command. The first violating substep is included, then the case stops.
    """
    torch.set_num_threads(1)
    if reference is None:
        reference = lambda t: globals()["reference"](t, speed, "coupled")
    model = BatchedUUVModel("cpu")
    mpc = CondensedActuatorMPC(model, 1, horizon=35, dt=PERIOD)
    mpc.dynamics = ActuatorAwareCpuDynamics(PERIOD, model_gain, model_tau)
    if mpc_setup is not None:
        mpc_setup(mpc)
    plant = Plant(speed, config)
    mpc.previous[:] = model.tensor(plant.actual/SCALE)
    mpc.plan[:] = mpc.previous[:, None]
    arrays = {key: [] for key in ("time", "state", "output", "reference", "error",
               "diagnostic", "actual", "subcommand", "command_time", "command",
               "qp_ok", "qp_primal", "qp_dual", "one_step_error")}
    first_violation = None
    start = time.perf_counter()
    for k in range(round(duration/PERIOD)):
        z = plant.controller_state()
        refs = reference((k+np.arange(1, 36))*PERIOD)
        command, ok = mpc.solve(model.tensor(z)[None], model.tensor(refs)[None])
        command = command[0].numpy()
        predicted = mpc.dynamics.evaluate(z[None, None], (command/SCALE)[None, None])[0][0, 0]
        arrays["command_time"].append(k*PERIOD)
        arrays["command"].append(command.copy())
        arrays["qp_ok"].append(bool(ok[0]))
        arrays["qp_primal"].append(float(mpc.last_primal_residual[0]))
        arrays["qp_dual"].append(float(mpc.last_dual_residual[0]))
        for j in range(5):
            plant.step(command)
            now = (5*k+j+1)*DT
            ref, output = reference(now), plant.output()
            error = ref-output
            error[3:5] = np.arctan2(np.sin(error[3:5]), np.cos(error[3:5]))
            diagnostics = diagnostic_values(output, ref)
            for key, value in (("time", now), ("state", plant.x.copy()),
                               ("output", output), ("reference", ref), ("error", error),
                               ("diagnostic", diagnostics), ("actual", plant.actual.copy()),
                               ("subcommand", command.copy())):
                arrays[key].append(value)
            exceeded = np.abs(diagnostics) > [8., 5., 2., np.deg2rad(40)]
            nonfinite = not np.isfinite(plant.x).all()
            if first_violation is None and (np.any(exceeded) or nonfinite):
                names = ["lateral_8m", "depth_5m", "tangent_speed_2mps", "pitch_40deg"]
                first_violation = {"time_s": now,
                    "kinds": [n for n, bad in zip(names, exceeded) if bad] + (["nonfinite"] if nonfinite else []),
                    "diagnostic_values": diagnostics.tolist()}
            if nonfinite or (stop_at_limit and first_violation is not None):
                break
        if j == 4:
            arrays["one_step_error"].append(float(np.max(np.abs(plant.controller_state()-predicted))))
        if not np.isfinite(plant.x).all() or (stop_at_limit and first_violation is not None):
            break
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    normalized = arrays["command"]/SCALE
    actual_normalized = arrays["actual"]/SCALE
    lower = np.array([0., -1., -1.])
    saturation = lambda values: np.mean((values >= 1-1e-6) | (values <= lower+1e-6), axis=0).tolist()
    meta = {"config": asdict(config), "speed_mps": speed, "duration_requested_s": duration,
            "duration_observed_s": float(arrays["time"][-1]), "substeps": len(arrays["time"]),
            "first_diagnostic_limit_violation": first_violation,
            "qp_failures": int(np.sum(~arrays["qp_ok"])),
            "command_saturation_fraction": saturation(normalized),
            "actual_saturation_fraction_substeps": saturation(actual_normalized),
            "max_actual_normalized_abs": np.max(np.abs(actual_normalized), axis=0).tolist(),
            "one_step_max_abs_state_error": float(np.max(arrays["one_step_error"])) if len(arrays["one_step_error"]) else None,
            "true_straight_trim_fraction": float(config.parameters[1]*D_COEFF[0, 0]*speed**2/config.parameters[2]/SCALE[0]),
            "model_gain": model_gain, "model_tau_scale": model_tau,
            "wall_s_not_realtime_benchmark": time.perf_counter()-start}
    return {"meta": meta, "arrays": arrays}


def parity_check():
    """Independent torch/RK4 and target actuator parity, varied true parameters."""
    from types import SimpleNamespace
    from aav_mpc.h15_target_domain import SecondOrderActuator
    model = BatchedUUVModel("cpu")
    rng = np.random.default_rng(9512)
    maximum = 0.
    for order in ("first", "second_all", "second_fins"):
        config = PlantConfig(mass=.9, damping=1.1, thrust=.9, elevator=.8,
                             rudder=1.1, delay_steps=(1, 5, 3),
                             tau_scale=(1.2, .5, 2.), order=order)
        plant = Plant(15., config)
        plant.x += np.array([0, 0, 0, .03, -.05, 0, .2, -.1, .01, -.02])
        x = model.tensor(plant.x)[None]
        env = SimpleNamespace(actuator=model.tensor(plant.actual)[None], tau=model.tensor(plant.tau)[None])
        second = SecondOrderActuator(env)
        for _ in range(30):
            command = SCALE*np.array([rng.uniform(.4, .8), rng.uniform(-.4, .4), rng.uniform(-.4, .4)])
            target = np.array([command[j] if delay == 0 else plant.queue[delay-1, j]
                               for j, delay in enumerate(config.delay_steps)])
            target = model.tensor(target)[None]
            first = env.actuator + (-torch.expm1(-DT/env.tau))*(target-env.actuator)
            if order == "first":
                env.actuator = first
            else:
                result = second(target, DT, torch.ones(1, dtype=torch.bool))
                env.actuator = result if order == "second_all" else torch.cat((first[:, :1], result[:, 1:]), -1)
                if order == "second_fins":
                    second.rate[:, 0] = 0.
            x = model.rk4(x, env.actuator, parameters=model.tensor(config.parameters)[None], dt=DT)
            plant.step(command)
            maximum = max(maximum, float(np.max(np.abs(x.numpy()[0]-plant.x))),
                          float(np.max(np.abs(env.actuator.numpy()[0]-plant.actual))))
    if maximum > 1e-9:
        raise AssertionError(f"plant parity failed: {maximum}")
    return {"maximum_absolute_difference": maximum, "seed": 9512,
            "trials": 90, "tolerance": 1e-9, "passed": True}
