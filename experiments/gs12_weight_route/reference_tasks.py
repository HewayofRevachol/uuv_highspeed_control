"""Frozen, analytic reference tasks for the GS-12 weight experiment."""
from __future__ import annotations

import numpy as np

T0 = 64.0


def _single_signal(t, k, phase, amplitude=0.006):
    """Same 16 s raised-cosine startup as the historical FRF protocol."""
    t = np.asarray(t, dtype=float)
    w = 2.0 * np.pi * k / T0
    raw = amplitude * np.sin(w * t + phase)
    draw = amplitude * w * np.cos(w * t + phase)
    g = np.where(t <= 0, 0.0, np.where(t < 16.0,
                    0.5 * (1.0 - np.cos(np.pi * t / 16.0)), 1.0))
    dg = np.where((t > 0) & (t < 16.0),
                  0.5 * np.pi / 16.0 * np.sin(np.pi * t / 16.0), 0.0)
    z, dz = g * raw, dg * raw + g * draw
    # Analytic integral of g(t) * amplitude*sin(w*t+phase).
    a = np.pi / 16.0
    def int_sin(x):
        return (np.cos(phase) - np.cos(w*x + phase)) / w
    def int_cos_sin(x):
        q = w-a
        second = x*np.sin(phase) if abs(q) <= 1e-12 else (1-np.cos(q*x+phase))/q
        return 0.5*((1-np.cos((w+a)*x+phase))/(w+a) + second)
    tc = np.clip(t, 0., 16.)
    during = 0.5 * amplitude * (int_sin(tc)-int_cos_sin(tc))
    at16 = 0.5 * amplitude * (int_sin(16.)-int_cos_sin(16.))
    tail = np.maximum(t-16., 0.)
    after = at16 + amplitude*(np.cos(w*16.+phase)-np.cos(w*(16.+tail)+phase))/w
    integ = np.where(t <= 0, 0., np.where(t < 16., during, after))
    return z, dz, integ


def single_frequency(t, k, axis, phase=0.0, amplitude=0.006, speed=15.0):
    """Return [N,E,D,pitch,yaw,VN,VE,VD], with exact position derivatives."""
    t = np.asarray(t, dtype=float)
    z, dz, iz = _single_signal(t, k, phase, amplitude)
    north, east, depth = speed*t, np.zeros_like(t), np.full_like(t, 20.)
    vn, ve, vd = np.full_like(t, speed), np.zeros_like(t), np.zeros_like(t)
    if axis == "east":
        east, ve = 8.*z, 8.*dz
    elif axis == "depth":
        depth, vd = 20.+5.*z, 5.*dz
    elif axis == "speed":
        north, vn = speed*t+2.*iz, speed+2.*z
    else:
        raise ValueError(f"unknown axis: {axis}")
    pitch = -np.arctan2(vd, np.hypot(vn, ve))
    yaw = np.arctan2(ve, vn)
    return np.stack((north, east, depth, pitch, yaw, vn, ve, vd), axis=-1)


def _smoothstep(t, center, width=2.0):
    """Cubic smooth transition from zero to one, with zero endpoint slopes."""
    u = np.clip((np.asarray(t, dtype=float)-(center-width/2))/width, 0., 1.)
    value = u*u*(3.-2.*u)
    deriv = np.where((u > 0.) & (u < 1.), 6.*u*(1.-u)/width, 0.)
    return value, deriv


def mixed_reference(t, phase=0.0, speed=15.0, switch_times=(20., 40.)):
    """Low→high→low depth/east reference with analytic blend derivatives."""
    t = np.asarray(t, dtype=float)
    q1, dq1 = _smoothstep(t, switch_times[0])
    q2, dq2 = _smoothstep(t, switch_times[1])
    high_mix = q1*(1.-q2)
    dhigh = dq1*(1.-q2)-q1*dq2
    low_mix = 1.-high_mix
    wlo, whi = 2*np.pi*8/T0, 2*np.pi*32/T0
    zlo = .006*np.sin(wlo*t+phase)
    dzlo = .006*wlo*np.cos(wlo*t+phase)
    zhi = .006*np.sin(whi*t+phase+.37)
    dzhi = .006*whi*np.cos(whi*t+phase+.37)
    z = low_mix*zlo+high_mix*zhi
    dz = low_mix*dzlo+high_mix*dzhi+dhigh*(zhi-zlo)
    east, ve = 8.*z, 8.*dz
    depth, vd = 20.+5.*z, 5.*dz
    vn, north = np.full_like(t, speed), speed*t
    pitch, yaw = -np.arctan2(vd, np.hypot(vn, ve)), np.arctan2(ve, vn)
    return np.stack((north, east, depth, pitch, yaw, vn, ve, vd), axis=-1)


def band_reference(t, band, axis, amplitude=.006, speed=15.0):
    """Historical 8-line B1/B4 signal and smooth startup, for 256 s FRF runs."""
    t = np.asarray(t, dtype=float)
    start = 1 if band == "B1" else (25 if band == "B4" else None)
    if start is None:
        raise ValueError("long protocol supports B1 or B4")
    bins = np.arange(start, start+8)
    phase = -np.pi*np.arange(8)*(np.arange(8)-1)/8
    z = np.zeros_like(t); dz = np.zeros_like(t); integ = np.zeros_like(t)
    for k, p in zip(bins, phase):
        a, b, c = _single_signal(t, int(k), float(p), amplitude)
        z += a; dz += b; integ += c
    north, east, depth = speed*t, np.zeros_like(t), np.full_like(t, 20.)
    vn, ve, vd = np.full_like(t, speed), np.zeros_like(t), np.zeros_like(t)
    if axis == "east": east, ve = 8.*z, 8.*dz
    elif axis == "depth": depth, vd = 20.+5.*z, 5.*dz
    elif axis == "speed": north, vn = speed*t+2.*integ, speed+2.*z
    elif axis != "zero": raise ValueError(f"unknown axis: {axis}")
    pitch, yaw = -np.arctan2(vd, np.hypot(vn, ve)), np.arctan2(ve, vn)
    return np.stack((north, east, depth, pitch, yaw, vn, ve, vd), axis=-1)


def task_reference(task, t):
    if task["kind"] == "single":
        return single_frequency(t, task["k"], task["axis"], task["phase_rad"],
                               task.get("amplitude", .006), task.get("speed_mps", 15.))
    if task["kind"] == "mixed":
        return mixed_reference(t, task["phase_rad"], task.get("speed_mps", 15.),
                               tuple(task.get("switch_times_s", (20., 40.))))
    if task["kind"] == "band":
        return band_reference(t, task["band"], task["axis"], task.get("amplitude", .006),
                              task.get("speed_mps", 15.))
    if task["kind"] == "zero":
        t = np.asarray(t, dtype=float); z = np.zeros_like(t); speed = task.get("speed_mps", 15.)
        return np.stack((speed*t, z, z+20., z, z, z+speed, z, z), axis=-1)
    raise ValueError(f"unknown task kind {task['kind']}")
