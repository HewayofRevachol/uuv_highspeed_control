"""78 causal features, with finite-window least-squares frequency projection."""
import torch

OBSERVATION_VERSION = 'gs22-78-v1'


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class Features:
    def __init__(self, dynamics):
        self.d = dynamics
        self.omega = dynamics.tensor([2*torch.pi/8, 6*torch.pi/8])
        t = torch.arange(80, device=dynamics.device, dtype=dynamics.dtype)*.1
        angles = t[:, None]*self.omega
        design = torch.stack((torch.ones_like(t), t, angles[:, 0].cos(),
                              angles[:, 0].sin(), angles[:, 1].cos(), angles[:, 1].sin()), -1)
        self.projection = torch.linalg.pinv(design)[2:]

    def frequency(self, signal, history=False):
        coefficients = torch.einsum('kt,bta->bak', self.projection, signal)
        real, imag = coefficients[..., ::2], -coefficients[..., 1::2]
        if history:
            angle = self.omega*7.9
            real, imag = real*angle.cos()-imag*angle.sin(), real*angle.sin()+imag*angle.cos()
        return torch.stack((real, imag), -1)

    def observe(self, env):
        d = self.d
        ref = env.references[:, env.k*5]
        x, output = env.x, d.output(env.x, env.actual)
        error = ref[:, :5]-x[:, :5]
        error[:, 3:5] = wrap(error[:, 3:5])
        motion = torch.cat((error/d.tensor([8., 8., 5., torch.pi/18, torch.pi/18]),
                            (x[:, 5:]-d.tensor([15., 0., 0., 0., 0.]))/d.tensor([3., 1., 1., .2, .2])), -1)
        current = torch.cat((ref[:, 3:5].sin(), ref[:, 3:5].cos()), -1)[:, [0, 2, 1, 3]]
        current = torch.cat((current, ref[:, 5:]/d.tensor([18., 1., 1.])), -1)
        actuator = torch.cat((env.actual/d.scale, env.command_history.flatten(1),
                              env.estimated_rate*d.plant_tau), -1)
        indices = env.k*5+torch.tensor([25, 50, 100], device=d.device)
        preview = ((env.references[:, indices][:, :, [1, 2, 5]]-ref[:, None, [1, 2, 5]]) /
                   d.tensor([.048, .030, .012])).flatten(1)
        dense = env.references[:, env.k*5:env.k*5+400:5, [1, 2, 5]]/d.tensor([8., 5., 2.])
        rf = self.frequency(dense)/.006
        # Buffers are newest-last after roll; all entries are measured at or before now.
        ef = self.frequency(env.error_history, history=True)
        ef = ef/d.tensor([.001/8, .001/5, .0001/2])[None, :, None, None]
        if env.k < 80:
            ef = torch.zeros_like(ef)
        count = min(env.k, 80)
        innovations = env.innovation_history[:, -count:] if count else env.innovation_history
        innovation = torch.cat((innovations.mean(1), innovations.square().mean(1).sqrt()), -1)
        innovation /= d.tensor([.01, .001, .001]*2)
        command = env.last_command
        margin = 2*torch.minimum(command-env.mpc.lower, env.mpc.upper-command)/(env.mpc.upper-env.mpc.lower)
        flags = torch.cat((margin, env.qp_ok[:, None].double(),
                           torch.full_like(env.beta[:, None], min(env.k/80, 1.))), -1)
        observation = torch.cat((motion, current, actuator,
                                 env.beta[:, None], env.previous_target[:, None], preview,
                                 rf.flatten(1), ef.flatten(1), innovation, flags), -1)
        return observation.clamp(-10, 10).float()
