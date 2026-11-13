"""Analytic position/velocity references; historical bridge has NO startup ramp."""
import numpy as np

from .common import CONFIG


def signal(t, k, phase, amplitude, ramp=True):
    t = np.asarray(t, dtype=float)
    w = 2 * np.pi * k / 64.
    raw = amplitude * np.sin(w*t + phase)
    derivative = amplitude * w * np.cos(w*t + phase)

    def integral(freq, x):
        if abs(freq) < 1e-12:
            return x * np.sin(phase)
        return (np.cos(phase) - np.cos(freq*x + phase)) / freq

    if not ramp:
        return raw, derivative, amplitude * integral(w, t)
    a = np.pi / 16.
    g = np.where(t < 16., .5*(1-np.cos(a*t)), 1.)
    dg = np.where(t < 16., .5*a*np.sin(a*t), 0.)
    tc = np.minimum(t, 16.)
    initial_integral = amplitude * (.5*integral(w, tc)
                                   -.25*integral(w+a, tc)-.25*integral(w-a, tc))
    tail = np.where(t > 16., amplitude*(np.cos(w*16+phase)-np.cos(w*t+phase))/w, 0.)
    return g*raw, g*derivative + dg*raw, initial_integral + tail


def blend(t, center, width=4.):
    s = np.clip((np.asarray(t)-center+width/2)/width, 0., 1.)
    return s*s*(3-2*s), np.where((s > 0) & (s < 1), 6*s*(1-s)/width, 0.)


def reference(task, t):
    t = np.asarray(t, dtype=float)
    speed = task['speed_mps']
    n, e, d = speed*t, np.zeros_like(t), np.full_like(t, 20.)
    vn, ve, vd = np.full_like(t, speed), np.zeros_like(t), np.zeros_like(t)
    kind = task['kind']
    axis = task.get('axis', 'zero')
    amplitude = task.get('amplitude', .006)
    phase = task.get('phase_rad', 0.)
    if kind == 'maneuver':
        maneuver = task['maneuver']
        if maneuver == 'straight':
            n = n + 1.2*(1-np.cos(.5*t))
            vn = vn + .6*np.sin(.5*t)
        if maneuver in ('horizontal', 'coupled'):
            e, ve = 6*(1-np.cos(.7*t)), 4.2*np.sin(.7*t)
        if maneuver in ('depth', 'coupled'):
            d, vd = 20+2.4*(1-np.cos(.84*t)), 2.016*np.sin(.84*t)
    elif axis != 'zero':
        def component(offset):
            if kind == 'bridge':
                bins = CONFIG['bins']
                phases = -np.pi*np.arange(len(bins))*(np.arange(len(bins))-1)/len(bins)
                values = [signal(t, k, p, amplitude, ramp=False) for k, p in zip(bins, phases)]
                return tuple(sum(value[j] for value in values) for j in range(3))
            if kind == 'single':
                return signal(t, task['k'], phase+offset, amplitude)
            # Switch paths use the same position amplitude at both frequencies.
            lo = signal(t, 8, phase+offset, amplitude)
            hi = signal(t, 24, phase+offset, amplitude)
            b1, db1 = blend(t, 48.)
            b2, db2 = blend(t, 96.)
            weight, dw = b1*(1-b2), db1*(1-b2)-b1*db2
            if task['order'] == 'high_low_high':
                weight, dw = 1-weight, -dw
            return ((1-weight)*lo[0]+weight*hi[0],
                    (1-weight)*lo[1]+weight*hi[1]+dw*(hi[0]-lo[0]), None)

        z, dz, iz = component(0.)
        if axis in ('east', 'coupled'):
            e, ve = 8*z, 8*dz
        if axis in ('depth', 'coupled'):
            zd, dzd, _ = component(task.get('depth_phase_offset', 0.))
            d, vd = 20+5*zd, 5*dzd
        if axis == 'speed':
            n, vn = n+2*iz, vn+2*z
    pitch = -np.arctan2(vd, np.hypot(vn, ve))
    yaw = np.arctan2(ve, vn)
    return np.stack((n, e, d, pitch, yaw, vn, ve, vd), axis=-1)


def frequency_stage(task, t):
    middle = 48 <= t < 96
    high = middle if task['order'] == 'low_high_low' else not middle
    return '24' if high else '8'
