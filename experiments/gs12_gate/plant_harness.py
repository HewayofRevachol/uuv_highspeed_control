"""GS-12 fixed-time reference and 20 ms plant, isolated from policy inputs."""
from __future__ import annotations

from dataclasses import asdict, replace
import math

import numpy as np

from simulator.batch_mpc import CONTROL_SCALE
from simulator.model import (D_COEFF, DT, MISMATCHED_PLANT, PlantConfig,
                             derivative, rotation_ned)


def plant_from_case(case):
    p = MISMATCHED_PLANT
    factors = case["plant"]
    return replace(p, mass_scale=p.mass_scale*factors.get("mass", 1.),
                   damping_scale=p.damping_scale*factors.get("damping", 1.),
                   thrust_gain=p.thrust_gain*factors.get("thrust", 1.),
                   elevator_gain=p.elevator_gain*factors.get("fin", 1.),
                   rudder_gain=p.rudder_gain*factors.get("fin", 1.),
                   thrust_tau_s=p.thrust_tau_s*factors.get("thrust_tau", factors.get("tau", 1.)),
                   fin_tau_s=p.fin_tau_s*factors.get("fin_tau", factors.get("tau", 1.)),
                   thrust_delay_s=DT*factors.get("delay", math.ceil(p.thrust_delay_s/DT)),
                   fin_delay_s=DT*factors.get("delay", math.ceil(p.fin_delay_s/DT)))


def trim_thrust(speed, plant):
    return plant.damping_scale*D_COEFF[0, 0]*speed*speed/plant.thrust_gain


def capacity_limited(case):
    return bool(trim_thrust(case["speed"], plant_from_case(case)) > CONTROL_SCALE[0])


class Reference:
    def __init__(self, case):
        self.case = case
        self.speed = float(case["speed"])
        self.duration = float(case["duration_s"])
        self.kind = case["maneuver"]
        self.amp = float(case.get("amplitude", 1.))
        self.omega = float(case.get("omega", .5))
        self.phase = float(case.get("phase", 0.))
        self.direction = np.asarray(case.get("direction", [1., 1., 1.]), float)

    def at(self, t):
        t = float(t)
        w, phase = self.omega, self.phase
        if self.kind == "frequency_direction":
            vector = np.asarray(self.case["complex_direction"], float)
            if vector.shape != (3, 2):
                raise ValueError("complex_direction requires three real/imag pairs")
            v = vector[:, 0]+1j*vector[:, 1]
            origin_time = float(self.case.get("frequency_origin_time_s", 0.))
            origin_north = float(self.case.get("frequency_origin_north_m", 0.))
            t_local = t-origin_time
            bins = self.case.get("multi_sine_bins")
            frequencies = ([int(k)*2*np.pi/64 for k in bins] if bins else [w])
            wave = sum(np.exp(1j*(wk*t_local+phase)) for wk in frequencies)
            integrated = sum((np.exp(1j*(wk*t_local+phase))-np.exp(1j*phase))/(1j*wk)
                             for wk in frequencies)
            derivative_wave = sum(1j*wk*np.exp(1j*(wk*t_local+phase)) for wk in frequencies)
            east = 8*self.amp*np.real(v[0]*wave)
            depth = 5*self.amp*np.real(v[1]*wave)
            vn = self.speed+2*self.amp*np.real(v[2]*wave)
            north = origin_north+self.speed*t_local+2*self.amp*np.real(v[2]*integrated)
            ve = 8*self.amp*np.real(v[0]*derivative_wave)
            vd = 5*self.amp*np.real(v[1]*derivative_wave)
            velocity = np.array([vn, ve, vd])
            pitch = -math.atan2(vd, math.hypot(vn, ve))
            yaw = math.atan2(ve, vn)
            return np.r_[north, east, depth, pitch, yaw, velocity]
        s = np.sin(w*t+phase)-np.sin(phase)
        c = (np.cos(phase)-np.cos(w*t+phase))/w-t*np.sin(phase)
        if self.kind == "straight":
            a = np.zeros(3)
        elif self.kind == "turn":
            a = np.array([0., 2.5, 0.])
        elif self.kind == "depth":
            a = np.array([0., 0., 1.5])
        elif self.kind == "speed":
            a = np.array([1.5, 0., 0.])
        elif self.kind == "coupled":
            a = np.array([1.0, 2.0, 1.2])
        elif self.kind == "direction":
            a = self.direction*np.array([1.2, 2.0, 1.2])
        else:
            raise ValueError(f"unknown maneuver: {self.kind}")
        a = self.amp*a
        velocity = np.array([self.speed, 0., 0.])+a*s
        position = np.array([self.speed*t, 0., 0.])+a*c
        pitch = -math.atan2(velocity[2], math.hypot(velocity[0], velocity[1]))
        yaw = math.atan2(velocity[1], velocity[0])
        return np.r_[position, pitch, yaw, velocity]

    def horizon(self, t, n=35, dt=.1):
        return np.stack([self.at(t+(i+1)*dt) for i in range(n)])


class PlantHarness:
    """True plant; only measurement() is passed to the controller."""
    def __init__(self, case):
        self.case = case
        self.config = plant_from_case(case)
        self.order = case["plant"].get("order", "first")
        self.reference = Reference(case)
        self.t = 0.
        self.state = np.zeros(10)
        self.state[:5] = self.reference.at(0)[:5]
        self.state[5:8] = rotation_ned(self.state[3], self.state[4]).T @ self.reference.at(0)[5:8]
        if self.reference.kind == "frequency_direction":
            self.state[:5] = 0.
            self.state[5:8] = [case["speed"], 0., 0.]
        self.state[:3] += np.asarray(case.get("initial_offset", [0., 0., 0.]), float)
        # Shared nominal trim is independent of the hidden true plant factors.
        self.trim = np.array([min(trim_thrust(case["speed"], MISMATCHED_PLANT),
                                  CONTROL_SCALE[0]), 0., 0.])
        self.actual = self.trim.copy()
        self.rate = np.zeros(3)
        self.queue = np.tile(self.trim, (5, 1))

    def measurement(self):
        # Only 100 ms hull state, measured actuator and issued-command queue.
        return {"t": self.t, "state": self.state.copy(), "actual": self.actual.copy(),
                "queue": self.queue.copy(), "reference": self.reference.at(self.t),
                "preview_0p5": self.reference.at(self.t+.5),
                "preview_1p0": self.reference.at(self.t+1.)}

    def step(self, issued, stop_when=None):
        issued = np.asarray(issued, float)
        if issued.shape != (3,) or not np.isfinite(issued).all():
            raise ValueError("physical issued command must be finite [3]")
        issued = np.clip(issued, [0., -CONTROL_SCALE[1], -CONTROL_SCALE[2]], CONTROL_SCALE)
        rows = []
        for _ in range(5):
            delays = [math.ceil(self.config.thrust_delay_s/DT-1e-12),
                      math.ceil(self.config.fin_delay_s/DT-1e-12),
                      math.ceil(self.config.fin_delay_s/DT-1e-12)]
            target = np.array([issued[j] if delays[j] == 0 else self.queue[delays[j]-1, j]
                               for j in range(3)])
            self.queue = np.vstack((issued, self.queue[:-1]))
            taus = [self.config.thrust_tau_s, self.config.fin_tau_s, self.config.fin_tau_s]
            for j in range(3):
                if (self.order == "second_all" or
                    (self.order == "second_fins" and j > 0) or
                    (self.order == "second_thrust" and j == 0)):
                    omega = 1/max(taus[j], .03)
                    self.rate[j] += DT*(omega*omega*(target[j]-self.actual[j])-2*omega*self.rate[j])
                    self.actual[j] += DT*self.rate[j]
                else:
                    self.actual[j] += -np.expm1(-DT/max(taus[j], 1e-9))*(target[j]-self.actual[j])
            # GS-12 target uses two midpoint integrations per plant substep.
            for _ in range(2):
                h = DT/2
                k1 = derivative(self.state, self.actual, self.config)
                k2 = derivative(self.state+h*k1/2, self.actual, self.config)
                self.state = self.state+h*k2
            self.t = round(self.t+DT, 10)
            rows.append({"t": self.t, "state": self.state.copy(),
                         "actual": self.actual.copy(), "issued": issued.copy(),
                         "reference": self.reference.at(self.t)})
            if stop_when is not None and stop_when(rows[-1]):
                break
        return rows


def make_case(case_id, split, speed, family, maneuver, seed, duration_s=24.,
              training=False):
    rng = np.random.default_rng(seed)
    plant = {"mass": 1., "damping": 1., "thrust": 1., "fin": 1.,
             "tau": 1., "delay": 3, "order": "first"}
    if family == "fin": plant["fin"] = float(rng.uniform(.9, 1.1) if training else .9)
    elif family == "mass": plant["mass"] = float(rng.uniform(.9, 1.1) if training else 1.1)
    elif family == "damping": plant["damping"] = float(rng.uniform(.9, 1.1) if training else 1.1)
    elif family == "thrust": plant["thrust"] = float(rng.uniform(.9, 1.1) if training else .9)
    elif family == "delay": plant["delay"] = int(rng.integers(1, 6) if training else 5)
    elif family == "tau": plant["tau"] = float(rng.uniform(.5, 2.) if training else 2.)
    elif family == "second_all": plant["order"] = "second_all"
    elif family == "second_fins": plant["order"] = "second_fins"
    elif family == "combination":
        plant.update(fin=float(rng.uniform(.9, 1.) if training else .9),
                     tau=float(rng.uniform(1., 2.) if training else 2.),
                     delay=int(rng.integers(1, 6) if training else 5),
                     order="second_all")
    elif family != "nominal": raise ValueError(family)
    case = {"case_id": case_id, "split": split, "seed": int(seed),
            "speed": float(speed), "family": family, "maneuver": maneuver,
            "duration_s": float(duration_s), "plant": plant,
            "amplitude": float(rng.uniform(.8, 1.1) if training else 1.),
            "omega": float(.5*rng.uniform(.9, 1.1) if training else .5),
            "phase": float(rng.uniform(0, 2*np.pi) if training else 0.),
            "initial_offset": [0., 0., 0.], "noise_seed": int(seed+10_000)}
    case["capacity_limited"] = capacity_limited(case)
    return case
