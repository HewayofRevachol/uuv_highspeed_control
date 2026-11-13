"""Frozen GS-22 task splits and analytic references (no plant information)."""
from itertools import product
import numpy as np


def high_weight(task, t):
    def blend(center):
        s = np.clip((np.asarray(t)-center+2)/4, 0., 1.)
        return s*s*(3-2*s), np.where((s > 0) & (s < 1), 6*s*(1-s)/4, 0.)
    a, da = blend(48.)
    b, db = blend(96.)
    h, dh = a*(1-b), da*(1-b)-a*db
    return (1-h, -dh) if task['order'] == 'high_low_high' else (h, dh)


def reference(task, t):
    t = np.asarray(t, dtype=float)
    speed, phase, amplitude = task['speed_mps'], task['phase_rad'], task['amplitude']
    h, dh = high_weight(task, t)
    g = np.where(t < 16, .5*(1-np.cos(np.pi*t/16)), 1.)
    dg = np.where(t < 16, np.pi/32*np.sin(np.pi*t/16), 0.)

    def component(offset):
        w = 2*np.pi/8
        lo, hi = np.sin(w*t+phase+offset), np.sin(3*w*t+phase+offset)
        raw = (1-h)*lo+h*hi
        derivative = (1-h)*w*np.cos(w*t+phase+offset) + h*3*w*np.cos(3*w*t+phase+offset)+dh*(hi-lo)
        return amplitude*g*raw, amplitude*(dg*raw+g*derivative)

    ze, ve = component(0.)
    zd, vd = component(np.pi/2 if task['axis'] == 'coupled' else 0.)
    east = 8*ze if task['axis'] == 'coupled' else np.zeros_like(t)
    ve = 8*ve if task['axis'] == 'coupled' else np.zeros_like(t)
    vd = 5*vd
    return np.stack((speed*t, east, 20+5*zd,
                     -np.arctan2(vd, np.hypot(speed, ve)), np.arctan2(ve, speed),
                     np.full_like(t, speed), ve, vd), -1)


def frozen_tasks():
    def task(split, i, plant, axis, order, phase, speed=15., amplitude=.006):
        return dict(id=f'{split}_{i:02d}', plant=plant, axis=axis, order=order,
                    phase_rad=phase, speed_mps=speed, amplitude=amplitude, duration_s=144.)
    orders = ('low_high_low', 'high_low_high')
    dev = [task('dev', i, 'combined', axis, order, .93)
           for i, (axis, order) in enumerate(product(('depth', 'coupled'), orders))]
    train = [task('train', i, plant, axis, order, phase)
             for i, (plant, axis, order, phase) in enumerate(product(
                 ('matched', 'combined'), ('depth', 'coupled'), orders, (.17, 1.73)))]
    holdout = [task('holdout', i, plant, axis, orders[i % 2], 2.41, amplitude=.003)
               for i, (plant, axis) in enumerate(product(('matched', 'combined'), ('depth', 'coupled')))]
    holdout += [task('holdout', i+4, 'combined', axis, orders[i % 2], 2.41, speed=speed)
                for i, (speed, axis) in enumerate(product((12., 18.), ('depth', 'coupled')))]
    return dict(dev=dev, train=train, holdout=holdout)
