"""Common exogenous GS-12 references for ten and twenty-two state runs."""
import numpy as np

T0=64.
SCALE=np.array([8.,5.,2.])

def envelope(t):
    t=np.asarray(t,float)
    g=np.where(t<=0,0.,np.where(t<16,.5*(1-np.cos(np.pi*t/16)),1.))
    gp=np.where((t>0)&(t<16),.5*np.pi/16*np.sin(np.pi*t/16),0.)
    gpp=np.where((t>0)&(t<16),np.pi**2/512*np.cos(np.pi*t/16),0.)
    return g,gp,gpp

def _probe(t,bins,phases,amp,window=True):
    t=np.asarray(t,float); bins=np.atleast_1d(bins); w=2*np.pi*bins/T0; phases=np.asarray(phases)
    ang=t[...,None]*w+phases
    raw=amp*np.sin(ang).sum(-1); raw1=amp*(np.cos(ang)*w).sum(-1)
    raw2=-amp*(np.sin(ang)*w*w).sum(-1)
    if window:
        g,gp,gpp=envelope(t)
        return g*raw, gp*raw+g*raw1, gpp*raw+2*gp*raw1+g*raw2
    return raw,raw1,raw2

def primary_phases(band):
    ix=int(band[1:])-1
    base=-np.pi*np.arange(8)*(np.arange(8)-1)/8
    return base

def multi_signal(t,band,amp=.006,phase="primary"):
    i=int(band[1:])-1; bins=np.arange(8*i+1,8*i+9)
    p=primary_phases(band)
    if phase=="secondary": p=p+np.random.default_rng(9512+i+1).uniform(-np.pi,np.pi,8)
    y,dy,_=_probe(t,bins,p,amp,True)
    integral=amp*sum(_integral_windowed_sine(t,2*np.pi*k/T0,ph) for k,ph in zip(bins,p))
    return y,dy,integral

def _pose(vn,ve,vd):
    yaw=np.arctan2(ve,vn); pitch=-np.arctan2(vd,np.hypot(vn,ve))
    return pitch,yaw

def _integral_windowed_sine(t,w,phase=0.):
    """Primitive from 0 to t of g(t) sin(w*t+phase), exact analytically."""
    t=np.asarray(t,float); a=np.pi/16.; tc=np.clip(t,0.,16.)
    def startup(x):
        one=(np.cos(phase)-np.cos(w*x+phase))/w
        plus=(np.cos(phase)-np.cos((w+a)*x+phase))/(w+a)
        if abs(w-a)<1e-12: minus=x*np.sin(phase)
        else: minus=(np.cos(phase)-np.cos((w-a)*x+phase))/(w-a)
        return .5*(one-.5*(plus+minus))
    pre=startup(16.)
    tail=np.maximum(t-16.,0.)
    post=pre+(np.cos(w*16.+phase)-np.cos(w*(16.+tail)+phase))/w
    return np.where(t<=0.,0.,np.where(t<16.,startup(tc),post))

def straight(speed,band=None,axis=-1,amp=0.,phase="primary"):
    def ref(t):
        t=np.asarray(t,float); z,dz,iz=(np.zeros_like(t),)*3 if band is None or axis<0 else multi_signal(t,band,amp,phase)
        n=speed*t+(2*iz if axis==2 else 0.)
        e=8*z if axis==0 else np.zeros_like(t); d=20+5*z if axis==1 else np.full_like(t,20.)
        vn=speed+2*z if axis==2 else np.full_like(t,speed)
        ve=8*dz if axis==0 else np.zeros_like(t); vd=5*dz if axis==1 else np.zeros_like(t)
        pitch,yaw=_pose(vn,ve,vd)
        return np.stack((n,e,d,pitch,yaw,vn,ve,vd),axis=-1)
    return ref

def turn(speed,curvature,band=None,axis=-1,amp=0.,phase="primary"):
    """Constant-radius local reference; axis 0 lateral, 1 depth, 2 speed."""
    k=float(curvature)
    def ref(t):
        t=np.asarray(t,float); z,dz,iz=(np.zeros_like(t),)*3 if band is None or axis<0 else multi_signal(t,band,amp,phase)
        s=speed*t+(2*iz if axis==2 else 0.); sd=speed+2*z if axis==2 else np.full_like(t,speed)
        psi=k*s; tangent=np.stack((np.cos(psi),np.sin(psi),np.zeros_like(t)),axis=-1)
        normal=np.stack((-np.sin(psi),np.cos(psi),np.zeros_like(t)),axis=-1)
        center=np.stack((np.sin(psi)/k,(1-np.cos(psi))/k,np.full_like(t,20.)),axis=-1)
        lateral=8*z if axis==0 else np.zeros_like(t); depth=5*z if axis==1 else np.zeros_like(t)
        pos=center+lateral[...,None]*normal+depth[...,None]*np.array([0.,0.,1.])
        dzl=8*dz if axis==0 else np.zeros_like(t)
        vel=sd[...,None]*(1-k*lateral)[...,None]*tangent+dzl[...,None]*normal
        if axis==1: vel=vel+np.stack((np.zeros_like(t),np.zeros_like(t),5*dz),axis=-1)
        vn,ve,vd=vel[...,0],vel[...,1],vel[...,2]; pitch,yaw=_pose(vn,ve,vd)
        return np.stack((pos[...,0],pos[...,1],pos[...,2],pitch,yaw,vn,ve,vd),axis=-1)
    return ref

def slope(speed,gamma,band=None,axis=-1,amp=0.,phase="primary"):
    def ref(t):
        t=np.asarray(t,float); z,dz,iz=(np.zeros_like(t),)*3 if band is None or axis<0 else multi_signal(t,band,amp,phase)
        s=speed*t+(2*iz if axis==2 else 0.); sd=speed+2*z if axis==2 else np.full_like(t,speed)
        tangent=np.array([np.cos(gamma),0.,np.sin(gamma)]); normal=np.array([0.,1.,0.]); depth=np.array([-np.sin(gamma),0.,np.cos(gamma)])
        zl=8*z if axis==0 else np.zeros_like(t); zd=5*z if axis==1 else np.zeros_like(t)
        zld=8*dz if axis==0 else np.zeros_like(t); zdd=5*dz if axis==1 else np.zeros_like(t)
        pos=s[...,None]*tangent+np.array([0.,0.,280.])+zl[...,None]*normal+zd[...,None]*depth
        vel=sd[...,None]*tangent+zld[...,None]*normal+zdd[...,None]*depth
        vn,ve,vd=vel[...,0],vel[...,1],vel[...,2];pitch,yaw=_pose(vn,ve,vd)
        return np.stack((pos[...,0],pos[...,1],pos[...,2],pitch,yaw,vn,ve,vd),axis=-1)
    return ref

def periodic(speed,kind,probe_k=None,axis=-1,amp=.006):
    """S-bend or coupled periodic trajectory and optional single-frequency probe."""
    A=15.*(speed/15.)**2; wb=2*np.pi/16.; ws=2*np.pi/64.
    def ref(t):
        t=np.asarray(t,float);g,gp,gpp=envelope(t)
        eb=A*g*np.sin(wb*t); eb1=A*(gp*np.sin(wb*t)+g*wb*np.cos(wb*t))
        db=np.full_like(t,20.); db1=np.zeros_like(t)
        vn0=np.full_like(t,speed); vn01=np.zeros_like(t)
        if kind=="coupled":
            db=20+2*g*np.sin(wb*t+np.pi/2); db1=2*(gp*np.sin(wb*t+np.pi/2)+g*wb*np.cos(wb*t+np.pi/2))
            vn0=speed+.5*g*np.sin(ws*t); vn01=.5*(gp*np.sin(ws*t)+g*ws*np.cos(ws*t))
        n0=speed*t
        # Exact integral of the startup-windowed speed oscillation is supplied below.
        if kind=="coupled":
            n0=speed*t+.5*_integral_windowed_sine(t,ws)
        if probe_k is not None and axis>=0:
            z,dz,_=_probe(t,[probe_k],[0.],amp,True)
            if axis==0: eb,eb1=eb+8*z,eb1+8*dz
            elif axis==1: db,db1=db+5*z,db1+5*dz
            else:
                vn0=vn0+2*z
                n0=n0+2*amp*_integral_windowed_sine(t,2*np.pi*probe_k/T0)
            if axis==0: pass
        pitch,yaw=_pose(vn0,eb1,db1)
        return np.stack((n0,eb,db,pitch,yaw,vn0,eb1,db1),axis=-1)
    return ref

def coordinate_errors(reference_values,output,regime,speed=None,curvature=0.,gamma=0.):
    """Return normalized error in each regime's fixed analysis coordinates."""
    r=np.asarray(reference_values);y=np.asarray(output)
    dp=r[...,:3]-y[...,:3];dv=r[...,5:8]-y[...,5:8]
    if regime=="turn":
        psi0=curvature*(speed*r[...,0]*0+speed) # placeholder overwritten using time by caller
        raise ValueError("turn coordinates require explicit time; use coordinate_errors_at_time")
    if regime=="slope":
        nvec=np.array([0.,1.,0.]);dvec=np.array([-np.sin(gamma),0.,np.cos(gamma)]);tvec=np.array([np.cos(gamma),0.,np.sin(gamma)])
        return np.stack((dp@nvec/8.,dp@dvec/5.,dv@tvec/2.),axis=-1)
    return np.stack((dp[...,1]/8.,dp[...,2]/5.,dv[...,0]/2.),axis=-1)

def turn_errors(t,reference_values,output,speed,curvature):
    t=np.asarray(t);psi=curvature*speed*t
    tangent=np.stack((np.cos(psi),np.sin(psi),np.zeros_like(t)),axis=-1)
    normal=np.stack((-np.sin(psi),np.cos(psi),np.zeros_like(t)),axis=-1)
    dp=np.asarray(reference_values)[...,:3]-np.asarray(output)[...,:3]
    dv=np.asarray(reference_values)[...,5:8]-np.asarray(output)[...,5:8]
    return np.stack((np.sum(normal*dp,axis=-1)/8.,dp[...,2]/5.,np.sum(tangent*dv,axis=-1)/2.),axis=-1)
