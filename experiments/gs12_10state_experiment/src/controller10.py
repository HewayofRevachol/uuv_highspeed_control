"""Fair 10-state condensed RTI MPC for the GS-12 frequency experiment."""
from functools import lru_cache

import numpy as np
import osqp
import scipy.sparse as sp
import time

from simulator.batch_mpc import CONTROL_SCALE
from simulator.model import D_COEFF, ELEVATOR_COEFF, MASS, M_EFF, RUDDER_COEFF
from simulator.model import MISMATCHED_PLANT as BASE


@lru_cache(maxsize=8)
def _function(dt, substeps, fin_gain):
    import casadi as ca

    x, c = ca.SX.sym("x", 10), ca.SX.sym("c", 3)
    inverse_mass = ca.DM(np.linalg.inv(BASE.mass_scale*M_EFF))
    damping, elevator, rudder = map(ca.DM, (D_COEFF, ELEVATOR_COEFF, RUDDER_COEFF))
    scale = ca.DM(CONTROL_SCALE)
    params = BASE

    def rhs(s):
        theta, psi = s[3], s[4]
        u, v, w, q, r = (s[i] for i in range(5, 10))
        ct, st, cp, sn = ca.cos(theta), ca.sin(theta), ca.cos(psi), ca.sin(psi)
        pd = ca.vertcat(cp*ct*u-sn*v+cp*st*w, sn*ct*u+cp*v+sn*st*w, -st*u+ct*w)
        coriolis = params.mass_scale*MASS*ca.vertcat(-r*v+q*w, r*u, -q*u, 0, 0)
        physical = ca.times(scale, c)
        force = ca.vertcat(params.thrust_gain*physical[0], 0, 0, 0, 0)
        force += u*u*fin_gain*(params.elevator_gain*elevator*physical[1]+
                              params.rudder_gain*rudder*physical[2])
        acc = inverse_mass@(force-coriolis-params.damping_scale*u*damping@s[5:10])
        return ca.vertcat(pd, q, r/ct, acc)

    h = dt/substeps
    y = x
    for _ in range(substeps):
        k1 = rhs(y); k2 = rhs(y+h*k1/2); k3 = rhs(y+h*k2/2); k4 = rhs(y+h*k3)
        y = y+h*(k1+2*k2+2*k3+k4)/6
    return ca.Function("gs12_f10", [x, c], [y, ca.jacobian(y, x), ca.jacobian(y, c)])


class Dynamics10:
    nx = 10
    def __init__(self, dt=.1, substeps=5, fin_gain=1.):
        self.dt, self.substeps, self.fin_gain = float(dt), int(substeps), float(fin_gain)
        self.function = _function(self.dt, self.substeps, self.fin_gain)

    def evaluate(self, states, controls):
        states, controls = np.asarray(states), np.asarray(controls)
        if states.shape[:-1] != controls.shape[:-1] or states.shape[-1] != 10 or controls.shape[-1] != 3:
            raise ValueError("states/controls require matching [...,10] and [...,3] shapes")
        shape = states.shape[:-1]; n = int(np.prod(shape))
        out = [self.function(states.reshape(n,10)[i], controls.reshape(n,3)[i]) for i in range(n)]
        return tuple(np.stack([np.asarray(v[j]).reshape((10,10) if j == 1 else (10,3) if j == 2 else (10,))
                               for v in out]).reshape(*shape, *((10,10) if j == 1 else (10,3) if j == 2 else (10,)))
                     for j in range(3))


class CondensedMPC10:
    """N=35, one RTI-QP, only 3N normalized command increments."""
    def __init__(self, horizon=35, dt=.1, fin_gain=1.):
        self.horizon, self.dt = int(horizon), float(dt)
        self.dynamics = Dynamics10(dt, 5, fin_gain)
        self.plan = np.tile([.6,0.,0.], (self.horizon,1))
        self.previous = np.array([.6,0.,0.])
        self.last = {}

    def reset(self, trim_command):
        self.previous = np.asarray(trim_command, dtype=float).copy()
        self.plan[:] = self.previous

    @staticmethod
    def output_map(x):
        th, ps = x[3:5]; u,v,w=x[5:8]
        ct,st,cp,sn=np.cos(th),np.sin(th),np.cos(ps),np.sin(ps)
        out=np.zeros((8,10)); out[:3,:3]=np.eye(3); out[3,3]=out[4,4]=1.
        out[5:8,3]=[-cp*st*u+cp*ct*w,-sn*st*u+sn*ct*w,-ct*u-st*w]
        out[5:8,4]=[-sn*ct*u-cp*v-sn*st*w,cp*ct*u-sn*v+cp*st*w,0.]
        out[5:8,5:8]=[[cp*ct,-sn,cp*st],[sn*ct,cp,sn*st],[-st,0.,ct]]
        vel=np.array([cp*ct*u-sn*v+cp*st*w,sn*ct*u+cp*v+sn*st*w,-st*u+ct*w])
        return out, np.r_[x[:5],vel]-out@x

    def solve(self, state, reference):
        solve_start=time.perf_counter();model_s=0.
        state=np.asarray(state,float); reference=np.asarray(reference,float)
        n=self.horizon; dim=3*n
        if state.shape!=(10,) or reference.shape!=(n,8): raise ValueError("invalid state/reference shape")
        plan=np.vstack((self.plan[1:],self.plan[-1])).copy()
        hess=np.eye(dim)*1e-6; grad=np.zeros(dim)
        iw=np.tile([.15,.08,.08],n); rw=np.tile([1.8,.35,.35],n)
        diff=np.eye(dim)
        for k in range(1,n): diff[3*k:3*k+3,3*k-3:3*k]=-np.eye(3)
        hess+=np.diag(iw)+diff.T@(rw[:,None]*diff)
        grad+=iw*plan.ravel()+diff.T@(rw*(diff@plan.ravel()-np.r_[self.previous,np.zeros(dim-3)]))
        trajectory=np.zeros((n+1,10));trajectory[0]=state
        influence=np.zeros((10,dim)); q=np.array([3.,3.,3.,100.,30.,5.,5.,5.])
        for k in range(n):
            t0=time.perf_counter()
            f,a,b=self.dynamics.evaluate(trajectory[k][None],plan[k][None])
            model_s+=time.perf_counter()-t0
            trajectory[k+1]=f[0]; influence=a[0]@influence; influence[:,3*k:3*k+3]+=b[0]
            c,offset=self.output_map(trajectory[k+1]); err=c@trajectory[k+1]+offset-reference[k]
            err[4]=np.arctan2(np.sin(err[4]),np.cos(err[4])); sens=c@influence
            weight=q*(2.5 if k==n-1 else 1.)
            hess+=sens.T@(weight[:,None]*sens); grad+=sens.T@(weight*err)
        lo=np.tile([0.,-1.,-1.],n)-plan.ravel(); hi=np.ones(dim)-plan.ravel()
        assembly_s=time.perf_counter()-solve_start-model_s
        solver=osqp.OSQP(); setup_start=time.perf_counter(); solver.setup(P=sp.csc_matrix((hess+hess.T)/2),q=grad,A=sp.eye(dim,format='csc'),
              l=lo,u=hi,verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=10000,polishing=False)
        setup_s=time.perf_counter()-setup_start; solve_qp_start=time.perf_counter()
        sol=solver.solve(); qp_s=time.perf_counter()-solve_qp_start; valid=sol.info.status_val in (1,2) and sol.x is not None and np.isfinite(sol.x).all()
        if valid:
            plan=np.clip(plan+sol.x.reshape(n,3),[0.,-1.,-1.],[1.,1.,1.]); self.plan=plan
            self.previous=plan[0].copy()
        self.last={"valid":bool(valid),"status":sol.info.status,"primal":float(sol.info.prim_res),
                   "dual":float(sol.info.dual_res),"iterations":int(sol.info.iter),
                   "solve_time_s":float(sol.info.solve_time),"model_propagation_s":model_s,
                   "qp_assembly_s":assembly_s,"osqp_setup_s":setup_s,"osqp_solve_wall_s":qp_s,
                   "controller_wall_s":time.perf_counter()-solve_start}
        return self.previous*CONTROL_SCALE, trajectory
