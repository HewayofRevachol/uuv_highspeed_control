"""References for the frozen GS-12 joint predictor/weight experiment."""
from __future__ import annotations

import numpy as np

from experiments.gs12_weight_route.reference_tasks import single_frequency, mixed_reference, task_reference


def joint_task_reference(task, t):
    if task["kind"] == "beta_switch":
        t = np.asarray(t, dtype=float)
        phase = float(task.get("phase_rad", 0.))
        amplitude_low = float(task.get("amplitude_low", task.get("amplitude", .024)))
        amplitude_high = float(task.get("amplitude_high", task.get("amplitude", .024)))
        speed = float(task.get("speed_mps", 15.))
        low = task_reference({"kind": "single", "axis": "east", "k": 8,
                              "phase_rad": phase, "amplitude": amplitude_low,
                              "speed_mps": speed}, t)
        high = task_reference({"kind": "single", "axis": "east", "k": 32,
                               "phase_rad": phase, "amplitude": amplitude_high,
                               "speed_mps": speed}, t)

        def blend(center):
            u = np.clip((t-(center-2.))/4., 0., 1.)
            value = u*u*(3.-2.*u)
            derivative = np.where((u > 0.) & (u < 1.), 6.*u*(1.-u)/4., 0.)
            return value, derivative

        q1, dq1 = blend(20.)
        q2, dq2 = blend(44.)
        high_weight = q1*(1.-q2)
        dweight = dq1*(1.-q2)-q1*dq2
        if task.get("order", "low_high_low") == "high_low_high":
            high_weight, dweight = 1.-high_weight, -dweight
        east = (1.-high_weight)*low[:, 1]+high_weight*high[:, 1]
        ve = ((1.-high_weight)*low[:, 6]+high_weight*high[:, 6]
              +dweight*(high[:, 1]-low[:, 1]))
        yaw = np.arctan2(ve, speed)
        return np.stack((speed*t, east, np.full_like(t, 20.), np.zeros_like(t),
                         yaw, np.full_like(t, speed), ve, np.zeros_like(t)), axis=-1)
    if task["kind"] != "mixed":
        return task_reference(task, t)
    t = np.asarray(t, dtype=float)
    speed = float(task.get("speed_mps", 15.))
    amp = float(task.get("amplitude", .006))
    s = amp / .006
    base = mixed_reference(t, task["phase_rad"], speed,
                           tuple(task.get("switch_times_s", (21., 41.))))
    e0, d0, ve0, vd0 = base[..., 1], base[..., 2], base[..., 6], base[..., 7]
    g = np.where(t <= 0, 0., np.where(t < 16., .5*(1.-np.cos(np.pi*t/16.)), 1.))
    gp = np.where((t > 0.) & (t < 16.), .5*np.pi/16.*np.sin(np.pi*t/16.), 0.)
    east = s*g*e0
    depth = 20. + s*g*(d0-20.)
    ve = s*(g*ve0 + gp*e0)
    vd = s*(g*vd0 + gp*(d0-20.))
    north = speed*t
    vn = np.full_like(t, speed)
    pitch = -np.arctan2(vd, np.hypot(vn, ve))
    yaw = np.arctan2(ve, vn)
    return np.stack((north, east, depth, pitch, yaw, vn, ve, vd), axis=-1)


def derivative_error(task, fn=joint_task_reference):
    t = np.linspace(0., float(task["duration_s"]), int(task["duration_s"]*1000)+1)
    ref = fn(task, t)
    dt = t[1]-t[0]
    dpos = np.gradient(ref[:, :3], dt, axis=0, edge_order=2)
    return float(np.max(np.abs(dpos[1:-1] - ref[1:-1, 5:8])))
