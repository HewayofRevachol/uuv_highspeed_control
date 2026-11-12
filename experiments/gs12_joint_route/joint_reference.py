"""References for the frozen GS-12 joint predictor/weight experiment."""
from __future__ import annotations

import numpy as np

from experiments.gs12_weight_route.reference_tasks import single_frequency, mixed_reference, task_reference


def joint_task_reference(task, t):
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
