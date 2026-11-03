"""Physical rudder step diagnostic at fixed initial speeds 12/15/18 m/s."""
import argparse
import json
from pathlib import Path

import numpy as np

from simulator.model import (DT, ActuatorModel, D_COEFF, MISMATCHED_PLANT,
                             derivative, rk4_step)


def evaluate(speed, fin_deg=5., duration_s=30.):
    plant = MISMATCHED_PLANT
    thrust = plant.damping_scale * D_COEFF[0, 0] * speed**2 / plant.thrust_gain
    angle = np.deg2rad(fin_deg)
    command = np.array([thrust, 0., angle])
    x = np.zeros(10)
    x[5] = speed
    actuator = ActuatorModel(plant, DT)
    actuator.reset([thrust, 0., 0.])
    zero = np.array([thrust, 0., 0.])
    gain = (derivative(x, command, plant)[9] - derivative(x, zero, plant)[9]) / angle
    n = round(duration_s/DT)
    rate, distance, fin = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        actual = actuator.step(command)
        before = x[:3].copy()
        x = rk4_step(x, actual, plant_params=plant)
        rate[k] = x[9]
        distance[k] = np.linalg.norm(x[:3]-before) + (distance[k-1] if k else 0.)
        fin[k] = actual[2]
    steady = float(np.mean(rate[-round(2./DT):]))
    threshold = .9*abs(steady)
    hits = np.flatnonzero(np.sign(steady)*rate >= threshold)
    hit = int(hits[0]) if len(hits) else None
    return {"initial_speed_mps": speed, "commanded_rudder_deg": fin_deg,
            "thrust_trim_mn": thrust/1e6,
            "instantaneous_yaw_acceleration_gain_per_rad": float(gain),
            "steady_yaw_rate_radps": steady,
            "yaw_rate_90_percent_time_s": None if hit is None else (hit+1)*DT,
            "yaw_rate_90_percent_distance_m": None if hit is None else float(distance[hit]),
            "actual_fin_90_percent_time_s": float((np.flatnonzero(fin >= .9*angle)[0]+1)*DT)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"plant": "fixed_mismatch", "integration_period_s": DT,
              "results": [evaluate(speed) for speed in (12., 15., 18.)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
