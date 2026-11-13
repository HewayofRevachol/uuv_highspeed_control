"""Development candidate informed by the observed SVD channel tradeoff.

Thrust: nominal second-order prediction with a causal measured-position rate
observer. Fins: conservative first-order prediction, gain .85 and tau x2.
The original QP is loaded privately via improved_controller; no source edits.
"""
import numpy as np
import casadi as ca

from .improved_controller import SecondOrderDynamics, make_second_order_controller


class HybridDynamics(SecondOrderDynamics):
    nx = 25

    def __init__(self, dt=.1, fin_gain_scale=.85, fin_tau_scale=2.):
        from simulator.model import MISMATCHED_PLANT as p, MASS, M_EFF, D_COEFF, ELEVATOR_COEFF, RUDDER_COEFF
        from simulator.batch_mpc import CONTROL_SCALE
        if not np.isfinite([dt,fin_gain_scale,fin_tau_scale]).all() or min(dt,fin_gain_scale,fin_tau_scale)<=0:
            raise ValueError("finite positive model parameters required")
        if abs(dt/.02-round(dt/.02))>1e-9:
            raise ValueError("dt must use whole 20ms steps")
        self.fin_gain_scale, self.fin_tau_scale = fin_gain_scale, fin_tau_scale
        self.fins_only = False
        z, command = ca.SX.sym("z",25), ca.SX.sym("command",3)
        theta, residual = ca.SX.sym("theta",2), ca.SX.sym("residual",5)
        x, actual, queue, thrust_rate = z[:10], z[10:13], z[13:22], z[22]
        im = ca.DM(np.linalg.inv(p.mass_scale*M_EFF))
        omega = 1./p.thrust_tau_s
        fin_alpha = 1-ca.exp(-.02/(p.fin_tau_s*theta[1]))

        def body(s,a):
            th,ps,u,v,w,q,r = [s[i] for i in (3,4,5,6,7,8,9)]
            ct,st,cp,sp = ca.cos(th),ca.sin(th),ca.cos(ps),ca.sin(ps)
            vel=ca.vertcat(cp*ct*u-sp*v+cp*st*w,sp*ct*u+cp*v+sp*st*w,-st*u+ct*w)
            force=ca.vertcat(p.thrust_gain*a[0],0,0,0,0)
            force+=u*u*theta[0]*(p.elevator_gain*ca.DM(ELEVATOR_COEFF)*a[1]
                               +p.rudder_gain*ca.DM(RUDDER_COEFF)*a[2])
            c=p.mass_scale*MASS*ca.vertcat(-r*v+q*w,r*u,-q*u,0,0)
            acc=im@(force-c-p.damping_scale*u*ca.DM(D_COEFF)@s[5:10])+residual
            return ca.vertcat(vel,q,r/ct,acc)

        for _ in range(round(dt/.02)):
            target=queue[6:9]
            queue=ca.vertcat(command,queue[:6])
            thrust_rate+=.02*(omega**2*(target[0]-actual[0])-2*omega*thrust_rate)
            actual=ca.vertcat(actual[0]+.02*thrust_rate,
                             actual[1:3]+fin_alpha*(target[1:3]-actual[1:3]))
            physical=ca.DM(CONTROL_SCALE)*actual
            k1=body(x,physical); k2=body(x+.01*k1,physical)
            k3=body(x+.01*k2,physical); k4=body(x+.02*k3,physical)
            x=x+.02*(k1+2*k2+2*k3+k4)/6
        end=ca.vertcat(x,actual,queue,thrust_rate,0,0)
        self.function=ca.Function("gs12_hybrid25",[z,command,theta,residual],
            [end,ca.jacobian(end,z),ca.jacobian(end,command)])


class ThrustRateObserver:
    """Only reads measured thrust positions, issued thrust and queued thrusts."""
    def __init__(self):
        from simulator.model import MISMATCHED_PLANT as p
        self.omega=1./p.thrust_tau_s
        h,om=.02,self.omega
        step=np.array([[1-h*h*om*om,h*(1-2*h*om)],[-h*om*om,1-2*h*om]])
        self.H=np.linalg.matrix_power(step,5)
        if abs(self.H[0,1])<1e-6:
            raise ValueError("thrust rate reconstruction is poorly conditioned")

    def estimate(self,previous_actual,current_actual,previous_queue,issued_command):
        a=float(np.asarray(previous_actual)[0]); rate=0.
        queue=np.asarray(previous_queue).reshape(3,3)[:,0].copy()
        command=float(np.asarray(issued_command)[0])
        for _ in range(5):
            target=queue[2]
            queue=np.r_[command,queue[:2]]
            rate+=.02*(self.omega**2*(target-a)-2*self.omega*rate)
            a+=.02*rate
        previous_rate=(float(np.asarray(current_actual)[0])-a)/self.H[0,1]
        return np.array([rate+self.H[1,1]*previous_rate,0.,0.])


def make_hybrid_controller(model,repo,horizon=35):
    controller=make_second_order_controller(model,repo,horizon,fin_gain_scale=.85,fin_tau_scale=2.)
    controller.dynamics=HybridDynamics()
    return controller


def validate_hybrid():
    """Independent mixed-structure Torch propagation and channel separation."""
    from plant_harness import Plant, PlantConfig, SCALE, REPO
    from simulator.batch_mpc import BatchedUUVModel
    from simulator.rti_symbolic import ActuatorAwareCpuDynamics
    from simulator.model import MISMATCHED_PLANT as p
    rng=np.random.default_rng(95124)
    model=BatchedUUVModel("cpu")
    hybrid=HybridDynamics()
    order=SecondOrderDynamics(.1,.85,2.)
    conservative=ActuatorAwareCpuDynamics(.1,.85,2.)
    observer=ThrustRateObserver()
    plant=Plant(15.,PlantConfig(elevator=.85,rudder=.85,tau_scale=(1.,2.,2.)))
    x=model.tensor(plant.x)[None]
    x[0,3:5]=model.tensor([.03,-.02])
    actual=plant.actual/SCALE
    queue=plant.queue/SCALE
    rate=.025
    propagation=observer_error=actuator_error=0.
    for _ in range(20):
        z=np.r_[x.numpy()[0],actual,queue.ravel(),rate,.7,-.4]
        command=np.array([rng.uniform(.4,.9),rng.uniform(-.3,.3),rng.uniform(-.3,.3)])
        expected=hybrid.evaluate(z[None,None],command[None,None])[0][0,0]
        s=order.evaluate(z[None,None],command[None,None])[0][0,0]
        c=conservative.evaluate(z[:22][None,None],command[None,None])[0][0,0]
        actuator_error=max(actuator_error,abs(expected[10]-s[10]),abs(expected[22]-s[22]),
                           float(np.max(np.abs(expected[11:13]-c[11:13]))))
        old_actual,old_queue=actual.copy(),queue.copy()
        for _ in range(5):
            target=queue[2].copy()
            queue=np.vstack((command,queue[:2]))
            rate+=.02*((target[0]-actual[0])/p.thrust_tau_s**2-2*rate/p.thrust_tau_s)
            actual[0]+=.02*rate
            actual[1:]+=-np.expm1(-.02/(2*p.fin_tau_s))*(target[1:]-actual[1:])
            x=model.rk4(x,model.tensor(actual*SCALE)[None],model.tensor(plant.parameters)[None],dt=.02)
        actual_z=np.r_[x.numpy()[0],actual,queue.ravel(),rate,0.,0.]
        propagation=max(propagation,float(np.max(np.abs(actual_z-expected))))
        estimate=observer.estimate(old_actual,actual,old_queue,command)
        observer_error=max(observer_error,float(np.max(np.abs(estimate-[rate,0,0]))))
    assert max(propagation,observer_error,actuator_error)<1e-9
    # For an isolated thrust maneuver, full propagation matches order25.
    isolated=Plant(15.,PlantConfig())
    z=np.r_[isolated.controller_state(),.03,0,0]; command=np.array([.8,0,0])
    h=hybrid.evaluate(z[None,None],command[None,None])[0][0,0]
    o=order.evaluate(z[None,None],command[None,None])[0][0,0]
    isolated_thrust=float(np.max(np.abs(h-o)))
    # Constant thrust and arbitrary fins: full hull propagation matches conservative22.
    z=np.r_[isolated.controller_state(),0,0,0]; command=np.array([z[10],.2,-.1])
    h=hybrid.evaluate(z[None,None],command[None,None])[0][0,0]
    c=conservative.evaluate(z[:22][None,None],command[None,None])[0][0,0]
    isolated_fins=float(np.max(np.abs(h[:22]-c)))
    assert max(isolated_thrust,isolated_fins)<1e-9
    f,a,b=hybrid.evaluate(z[None,None],command[None,None])
    jacobian_error=0.; eps=1e-5
    for i in range(28):
        dz=np.zeros(25);du=np.zeros(3)
        if i<25:dz[i]=eps
        else:du[i-25]=eps
        plus=hybrid.evaluate((z+dz)[None,None],(command+du)[None,None])[0][0,0]
        minus=hybrid.evaluate((z-dz)[None,None],(command-du)[None,None])[0][0,0]
        expected=a[0,0,:,i] if i<25 else b[0,0,:,i-25]
        jacobian_error=max(jacobian_error,float(np.max(np.abs((plus-minus)/(2*eps)-expected))))
    assert jacobian_error<2e-7
    assert np.max(np.abs(a[0,0,:,23:25]))==0
    controller=make_hybrid_controller(model,REPO)
    assert controller.states.shape==(1,36,25)
    from aav_mpc import condensed_mpc
    assert condensed_mpc.AUGMENTED_NX==22
    return dict(passed=True,mixed_transition_max_abs=propagation,observer_max_abs=observer_error,
                actuator_subsystem_max_abs=actuator_error,isolated_thrust_order25_max_abs=isolated_thrust,
                isolated_fins_conservative22_max_abs=isolated_fins,jacobian_max_abs=jacobian_error,
                unused_fin_rate_jacobian_zero=True,production_dimension_unchanged=True,seed=95124)
