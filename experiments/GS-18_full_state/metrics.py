"""Offline ten-state reference and score for the GS-18 experiment."""
from __future__ import annotations

import numpy as np

from experiments.gs12_joint_route.joint_reference import joint_task_reference


DT = 0.02
N_STEPS = 3200
NAMES = ("N", "E", "D", "theta", "psi", "u", "v", "w", "q", "r")
SCALES = np.array([8., 8., 5., np.deg2rad(10.), np.deg2rad(10.),
                   2., 1., 1., np.deg2rad(5.), np.deg2rad(5.)])
GROUPS = {"position": (slice(0, 3), .40),
          "attitude": (slice(3, 5), .20),
          "body_velocity": (slice(5, 8), .25),
          "angular_rate": (slice(8, 10), .15)}


def full_reference(task: dict) -> np.ndarray:
    """Convert the existing eight-channel inertial reference to ten body states."""
    times = np.arange(N_STEPS + 1) * DT
    ref = joint_task_reference(task, times)
    theta, psi = ref[:, 3], np.unwrap(ref[:, 4])
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    vn, ve, vd = ref[:, 5], ref[:, 6], ref[:, 7]
    u = cp*ct*vn + sp*ct*ve - st*vd
    v = -sp*vn + cp*ve
    w = cp*st*vn + sp*st*ve + ct*vd
    q = np.gradient(theta, DT, edge_order=2)
    r = ct*np.gradient(psi, DT, edge_order=2)
    return np.column_stack((ref[:, :5], u, v, w, q, r))


def inertial_velocity(x: np.ndarray) -> np.ndarray:
    theta, psi = x[:, 3], x[:, 4]
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    u, v, w = x[:, 5], x[:, 6], x[:, 7]
    return np.column_stack((cp*ct*u-sp*v+cp*st*w,
                            sp*ct*u+cp*v+sp*st*w, -st*u+ct*w))


def score(x: np.ndarray, reference: np.ndarray, status: str) -> dict:
    """A failed run has diagnostics but no complete-episode quality score."""
    if not status.startswith("complete"):
        return {"J10": None, "J_track_recomputed": None,
                "groups": None, "rmse": None, "max_abs": None}
    error = x[1:] - reference[1:]
    error[:, 4] = np.arctan2(np.sin(error[:, 4]), np.cos(error[:, 4]))
    groups = {name: float((error[:, part]/SCALES[part]).__pow__(2).mean())
              for name, (part, _) in GROUPS.items()}
    j10 = sum(weight*groups[name] for name, (_, weight) in GROUPS.items())
    # Existing GS-12 window includes the sample at exactly 32 s.
    old_ref = joint_task_reference_from_full(reference)
    old_error = np.column_stack((old_ref[1600:, 1]-x[1600:, 1],
                                 old_ref[1600:, 2]-x[1600:, 2],
                                 old_ref[1600:, 5]-inertial_velocity(x[1600:])[:, 0]))
    old_j = float(np.mean(np.sum((old_error/np.array([8., 5., 2.]))**2, axis=1)))
    return {"J10": float(j10), "J_track_recomputed": old_j,
            "groups": groups,
            "rmse": {name: float(value) for name, value in zip(NAMES, np.sqrt(np.mean(error**2, axis=0)))},
            "max_abs": {name: float(value) for name, value in zip(NAMES, np.max(np.abs(error), axis=0))}}


def joint_task_reference_from_full(reference: np.ndarray) -> np.ndarray:
    """Recover inertial velocity to audit the historical GS-12 score."""
    return np.column_stack((reference[:, :5], inertial_velocity(reference)))
