"""GS-12 predictor and second-order research plant in batched torch.float64."""
from __future__ import annotations

import numpy as np
import torch

from simulator.batch_mpc import CONTROL_SCALE
from simulator.model import (D_COEFF, ELEVATOR_COEFF, MASS, M_EFF,
                             MISMATCHED_PLANT, RUDDER_COEFF)

DT = .02


class Dynamics:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.scale = self.tensor(CONTROL_SCALE)
        self.inv_mass = self.tensor(np.linalg.inv(M_EFF))
        self.damping = self.tensor(D_COEFF)
        self.elevator = self.tensor(ELEVATOR_COEFF)
        self.rudder = self.tensor(RUDDER_COEFF)
        p = MISMATCHED_PLANT
        self.plant_parameters = self.tensor([
            p.mass_scale, p.damping_scale, p.thrust_gain,
            .9*p.elevator_gain, .9*p.rudder_gain])
        self.plant_tau = self.tensor([p.thrust_tau_s, p.fin_tau_s, p.fin_tau_s])
        self.predictor_parameters = self.tensor([
            p.mass_scale, p.damping_scale, p.thrust_gain,
            p.elevator_gain, p.rudder_gain])

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def hull_rhs(self, x, physical, parameters):
        theta, psi = x[..., 3], x[..., 4]
        u, v, w, q, r = x[..., 5:].unbind(-1)
        ct, st, cp, sp = torch.cos(theta), torch.sin(theta), torch.cos(psi), torch.sin(psi)
        position = torch.stack((cp*ct*u-sp*v+cp*st*w,
                                sp*ct*u+cp*v+sp*st*w, -st*u+ct*w), -1)
        coriolis = parameters[..., 0, None]*MASS*torch.stack(
            (-r*v+q*w, r*u, -q*u, torch.zeros_like(u), torch.zeros_like(u)), -1)
        force = torch.stack((parameters[..., 2]*physical[..., 0],
                             torch.zeros_like(u), torch.zeros_like(u),
                             torch.zeros_like(u), torch.zeros_like(u)), -1)
        force = force + u[..., None].square()*(
            parameters[..., 3, None]*physical[..., 1, None]*self.elevator +
            parameters[..., 4, None]*physical[..., 2, None]*self.rudder)
        damp = parameters[..., 1, None]*u[..., None]*(x[..., 5:] @ self.damping.T)
        acceleration = ((force-coriolis-damp) @ self.inv_mass.T)/parameters[..., 0, None]
        return torch.cat((position, q[..., None], (r/ct)[..., None], acceleration), -1)

    def hull_step(self, x, physical, parameters):
        f = lambda y: self.hull_rhs(y, physical, parameters)
        k1 = f(x)
        k2 = f(x + .5*DT*k1)
        k3 = f(x + .5*DT*k2)
        k4 = f(x + DT*k3)
        return x + (DT/6.)*(k1+2*k2+2*k3+k4)

    def output(self, x, actual):
        velocity = self.hull_rhs(x, actual, self.plant_parameters)[..., :3]
        return torch.cat((x[..., :5], velocity), -1)

    def predictor_step(self, z, command, beta):
        """The first-order 22-state MPC model; beta is lane-specific."""
        x, actual, queue = z[..., :10], z[..., 10:13], z[..., 13:].reshape(*z.shape[:-1], 3, 3)
        p = MISMATCHED_PLANT
        parameters = self.predictor_parameters.expand(*beta.shape, 5).clone()
        gain = 1. - .15*beta
        parameters = torch.cat((parameters[..., :3],
                                parameters[..., 3:]*gain[..., None]), -1)
        tau = torch.stack((torch.full_like(beta, p.thrust_tau_s),
                           p.fin_tau_s*(1.+beta), p.fin_tau_s*(1.+beta)), -1)
        rate = -torch.expm1(-DT/tau)
        for _ in range(5):
            target = queue[..., 2, :]
            queue = torch.cat((command[..., None, :], queue[..., :-1, :]), -2)
            actual = actual + rate*(target-actual)
            x = self.hull_step(x, actual*self.scale, parameters)
        return torch.cat((x, actual, queue.reshape(*z.shape[:-1], 9)), -1)

    def plant_step(self, x, actual, rate, queue, command):
        """One real 0.02-second substep with the second_all actuator."""
        target = queue[..., 2, :]
        queue = torch.cat((command[..., None, :], queue[..., :-1, :]), -2)
        omega = 1./torch.maximum(self.plant_tau, self.tensor(.03))
        rate = rate + DT*(omega.square()*(target-actual)-2.*omega*rate)
        actual = actual + DT*rate
        x = self.hull_step(x, actual, self.plant_parameters)
        return x, actual, rate, queue

    def initial(self, speeds):
        speeds = self.tensor(speeds)
        trim = (MISMATCHED_PLANT.damping_scale*float(D_COEFF[0, 0]) /
                MISMATCHED_PLANT.thrust_gain)*speeds.square()
        zeros = torch.zeros_like(speeds)
        x = torch.stack((zeros, zeros, torch.full_like(speeds, 20.), zeros,
                         zeros, speeds, zeros, zeros, zeros, zeros), -1)
        actual = torch.stack((trim, zeros, zeros), -1)
        queue = actual[:, None, :].expand(-1, 3, -1).clone()
        rate = torch.zeros_like(actual)
        return x, actual, rate, queue
