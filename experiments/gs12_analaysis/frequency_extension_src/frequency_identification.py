"""Nonlinear closed-loop empirical reference-error FRFs for GS-12.

Independent physical reference generators, fixed future preview. No production edits.
"""
import os
for _name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'uuv_highspeed_control'))
SCALE=np.array([8.,5.,2.])
BINS=np.array([2,4,6,8,12,16,24])
OMEGA=2*np.pi*BINS/64.
PHASE=-np.pi*np.arange(len(BINS))*(np.arange(len(BINS))-1)/len(BINS)
SETTLE=16.
PERIOD=64.
DT=.1
CONFIGS={
 'matched':dict(gain=1.,tau=1.,delay=3,second=False),
 'fin_0p9':dict(gain=.9,tau=1.,delay=3,second=False),
 'delay5':dict(gain=1.,tau=1.,delay=5,second=False),
 'tau2':dict(gain=1.,tau=2.,tau_scope='all',delay=3,second=False),
 'fin_tau2':dict(gain=1.,tau=2.,tau_scope='fins',delay=3,second=False),
 'second_order':dict(gain=1.,tau=1.,delay=3,second=True),
 'combined':dict(gain=.9,tau=1.,delay=3,second=True),
}

def reference(t,speed,axis,amplitude=.006):
    t=np.asarray(t)
    angles=t[...,None]*OMEGA+PHASE
    signal=amplitude*np.sin(angles).sum(axis=-1)
    rate=amplitude*(OMEGA*np.cos(angles)).sum(axis=-1)
    integral=amplitude*((np.cos(PHASE)-np.cos(angles))/OMEGA).sum(axis=-1)
    north=speed*t
    east=np.zeros_like(t); depth=np.full_like(t,20.)
    vn=np.full_like(t,speed); ve=np.zeros_like(t); vd=np.zeros_like(t)
    if axis==0: east=8*signal; ve=8*rate
    elif axis==1: depth=depth+5*signal; vd=5*rate
    elif axis==2: vn=vn+2*signal; north=north+2*integral
    pitch=-np.arctan2(vd,np.sqrt(vn**2+ve**2))
    yaw=np.arctan2(ve,vn)
    return np.stack((north,east,depth,pitch,yaw,vn,ve,vd),axis=-1)

def filename(speed,config,axis,amplitude):
    return f'v{speed:g}_{config}_axis{axis}_a{amplitude:g}'

def run_case(task):
    speed,config,axis,amplitude,out=task
    tag=filename(speed,config,axis,amplitude)
    target=Path(out)/(tag+'.npz')
    meta_path=Path(out)/(tag+'.json')
    if target.exists() and meta_path.exists():
        return json.loads(meta_path.read_text())
    from simulator.batch_mpc import BatchedUUVModel
    from simulator.model import MISMATCHED_PLANT as p,D_COEFF
    from aav_mpc.condensed_mpc import CondensedActuatorMPC
    from aav_mpc.h15_target_domain import SecondOrderActuator
    from types import SimpleNamespace
    torch.set_num_threads(1)
    started=time.perf_counter()
    cfg=CONFIGS[config]
    model=BatchedUUVModel('cpu')
    mpc=CondensedActuatorMPC(model,1,horizon=35,dt=.1)
    x=model.tensor([[0.,0.,20.,0.,0.,speed,0.,0.,0.,0.]])
    trim=p.damping_scale*D_COEFF[0,0]*speed**2/p.thrust_gain
    initial=model.tensor([[trim,0.,0.]])
    plant=SimpleNamespace(actuator=initial.clone(),tau=model.tensor([[p.thrust_tau_s*(cfg['tau'] if cfg.get('tau_scope')=='all' else 1.),p.fin_tau_s*cfg['tau'],p.fin_tau_s*cfg['tau']]]))
    actuator_update=SecondOrderActuator(plant) if cfg['second'] else None
    queue=initial[:,None].repeat(1,max(3,cfg['delay']),1)
    parameters=model.tensor([[p.mass_scale,p.damping_scale,p.thrust_gain,p.elevator_gain*cfg['gain'],p.rudder_gain*cfg['gain']]])
    mpc.previous[:]=initial/model.control_scale
    mpc.plan[:]=mpc.previous[:,None]
    active=torch.ones(1,dtype=torch.bool)
    n=round((SETTLE+2*PERIOD)/DT)
    times=(np.arange(n)+1)*DT
    refs=reference(times,speed,axis,amplitude)
    all_preview=reference((np.arange(n+35)+1)*DT,speed,axis,amplitude)
    states=np.empty((n,10));commands=np.empty((n,3));actuators=np.empty((n,3));outputs=np.empty((n,8));okays=np.empty(n,dtype=bool)
    parity=0.
    for k in range(n):
        z=torch.cat((x,plant.actuator/model.control_scale,queue[:,:3].reshape(1,9)/model.control_scale.repeat(3)),dim=-1)
        cmd,ok=mpc.solve(z,model.tensor(all_preview[k:k+35])[None])
        if config=='matched':
            predicted=mpc.dynamics.evaluate(z.numpy()[:,None],(cmd/model.control_scale).numpy()[:,None])[0][0,0]
        for _ in range(5):
            delayed=queue[:,cfg['delay']-1].clone()
            queue=torch.cat((cmd[:,None],queue[:,:-1]),dim=1)
            if actuator_update is None:
                plant.actuator+=-torch.expm1(-.02/plant.tau)*(delayed-plant.actuator)
            else:
                plant.actuator=actuator_update(delayed,.02,active)
            x=model.rk4(x,plant.actuator,parameters=parameters,dt=.02)
        if config=='matched':
            actual=torch.cat((x,plant.actuator/model.control_scale,queue[:,:3].reshape(1,9)/model.control_scale.repeat(3)),dim=-1).numpy()[0]
            parity=max(parity,float(np.max(np.abs(predicted-actual))))
        vel=model.derivative(x,plant.actuator,parameters=parameters)[0,:3].numpy()
        states[k]=x[0].numpy();commands[k]=cmd[0].numpy();actuators[k]=plant.actuator[0].numpy();outputs[k]=np.r_[x[0,:5].numpy(),vel];okays[k]=bool(ok[0])
        if not np.isfinite(states[k]).all(): raise ValueError(f'Nonfinite {tag} step{k}')
    errors=(refs[:,[1,2,5]]-outputs[:,[1,2,5]])/SCALE
    inputs=np.column_stack((refs[:,1],refs[:,2]-20,refs[:,5]-speed))/SCALE
    normcmd=commands/model.control_scale.numpy()
    saturation=(normcmd>1-1e-6)|(normcmd<np.array([0.,-1.,-1.])+1e-6)
    np.savez_compressed(target,time_s=times,reference=refs,state=states,output=outputs,error_normalized=errors,input_normalized=inputs,command=commands,actuator=actuators,qp_ok=okays,omega_radps=OMEGA,scale=SCALE)
    meta=dict(tag=tag,speed=speed,config=config,axis=axis,amplitude_per_line_normalized=amplitude,plant=cfg,qp_failures=int((~okays).sum()),saturation_fraction=saturation.mean(axis=0).tolist(),max_abs_pitch_deg=float(np.rad2deg(np.abs(states[:,3]).max())),max_abs_fin_deg=np.rad2deg(np.abs(commands[:,1:]).max(axis=0)).tolist(),max_thrust_fraction=float(normcmd[:,0].max()),matched_one_step_max_abs=parity if config=='matched' else None,seconds=time.perf_counter()-started,filename=target.name)
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    return meta

def complex_dump(x): return dict(real=np.asarray(x).real.tolist(),imag=np.asarray(x).imag.tolist())

def estimate(out):
    out=Path(out)
    metas=[json.loads(p.read_text()) for p in out.glob('v*.json')]
    groups={}
    for m in metas:
        if m['axis']>=0: groups.setdefault((m['speed'],m['config'],m['amplitude_per_line_normalized']),{})[m['axis']]=m
    results=[]
    for (speed,config,amplitude),axes in sorted(groups.items()):
        if len(axes)<3: continue
        h=np.zeros((2,len(OMEGA),3,3),complex)
        distortion=[]
        baseline=out/(filename(speed,config,-1,0.)+'.npz')
        baseline_err=np.load(baseline)['error_normalized'] if baseline.exists() else None
        for axis,m in sorted(axes.items()):
            d=np.load(out/m['filename']); t=d['time_s'];e=d['error_normalized'];u=d['input_normalized'][:,axis]
            if baseline_err is not None: e=e-baseline_err
            for period in range(2):
                mask=(t>SETTLE+period*PERIOD+1e-8)&(t<=SETTLE+(period+1)*PERIOD+1e-8)
                tt=t[mask];ee=e[mask];uu=u[mask]
                ph=np.exp(-1j*tt[:,None]*OMEGA)
                E=2/len(tt)*ph.T@ee
                U=2/len(tt)*ph.T@uu
                h[period,:,:,axis]=E/U[:,None]
                if period==1:
                    residual=ee-ee.mean(axis=0)-np.real(np.exp(1j*tt[:,None]*OMEGA)@E)
                    distortion.append(float(np.linalg.norm(residual)/max(np.linalg.norm(ee-ee.mean(axis=0)),1e-15)))
        H=h[1]
        U,s,Vh=np.linalg.svd(H)
        v=Vh.conj().transpose(0,2,1)[:,:,0]
        # Common phase is immaterial. Largest component made positive real.
        for k in range(len(v)):
            v[k]*=np.exp(-1j*np.angle(v[k,np.argmax(np.abs(v[k]))]))
        left_max=np.einsum('kij,kj->ki',H,v)/s[:,0,None]
        convergence=np.linalg.norm(h[1]-h[0],axis=(1,2))/np.maximum(np.linalg.norm(h[1],axis=(1,2)),1e-12)
        result=dict(speed=speed,config=config,amplitude_per_line_normalized=amplitude,omega_radps=OMEGA.tolist(),H=complex_dump(H),H_period1=complex_dump(h[0]),singular_values=s.tolist(),right_max_vector=complex_dump(v),left_max_vector=complex_dump(left_max),right_max_vector_abs=np.abs(v).tolist(),right_max_vector_phase_deg=np.rad2deg(np.angle(v)).tolist(),period_relative_change=convergence.tolist(),unexcited_output_rms_fraction_by_input=distortion,baseline_removed=baseline_err is not None,baseline_mean_normalized=(baseline_err[-640:].mean(axis=0).tolist() if baseline_err is not None else None),peak_index=int(np.argmax(s[:,0])),peak_sigma=float(s[:,0].max()))
        np.savez_compressed(out/f'frf_v{speed:g}_{config}_a{amplitude:g}.npz',H=H,H_period1=h[0],omega_radps=OMEGA,singular_values=s,right_max_vector=v,left_max_vector=left_max,period_relative_change=convergence)
        results.append(result)
    (out/'frequency_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'gs12_followup/frequency_data');ap.add_argument('--speeds',nargs='+',type=float,default=[15]);ap.add_argument('--configs',nargs='+',default=list(CONFIGS));ap.add_argument('--amplitude',type=float,default=.006);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--baseline',action='store_true');ap.add_argument('--estimate-only',action='store_true');args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    if not args.estimate_only:
        tasks=[]
        for speed in args.speeds:
            for cfg in args.configs:
                if args.baseline:tasks.append((speed,cfg,-1,0.,str(args.out)))
                tasks.extend((speed,cfg,axis,args.amplitude,str(args.out)) for axis in range(3))
        with ProcessPoolExecutor(args.workers) as pool:
            fs={pool.submit(run_case,t):t for t in tasks}
            for f in as_completed(fs):
                print(json.dumps(f.result()),flush=True)
    results=estimate(args.out)
    for r in results: print(json.dumps({k:r[k] for k in ['speed','config','amplitude_per_line_normalized','peak_index','peak_sigma','period_relative_change','unexcited_output_rms_fraction_by_input']}),flush=True)

if __name__=='__main__':main()
