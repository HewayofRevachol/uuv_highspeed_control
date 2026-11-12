"""Small, explicit reference-spectrum feature and R0/R1 beta policies."""
import numpy as np
N=160;DT=.1;OMEGA_CUT=2.2;ENERGY_FLOOR=1e-8
def features(reference_window):
 x=np.asarray(reference_window,dtype=float)
 if x.shape!=(N,8):raise ValueError(f"expected {(N,8)} reference preview samples, got {x.shape}")
 z=np.column_stack((x[:,1]/8.,(x[:,2]-20.)/5.,(x[:,5]-15.)/2.))
 tt=np.arange(N,dtype=float); design=np.column_stack((np.ones(N),tt)); detr=z-design@np.linalg.lstsq(design,z,rcond=None)[0]
 spec=np.fft.rfft(detr*np.hanning(N)[:,None],axis=0); power=np.abs(spec)**2/(N*N); omega=2*np.pi*np.fft.rfftfreq(N,d=DT)
 keep=omega>0; total=float(power[keep].sum()); high=float(power[omega>=OMEGA_CUT].sum())
 return {"E_total":total,"rho_high":(high/total if total>=ENERGY_FLOOR else None),"low_energy":bool(total<ENERGY_FLOOR),"omega_resolution_rad_s":float(2*np.pi/(N*DT))}

class SpectralBetaPolicy:
 def __init__(self,initial_beta=.75,policy="R0"):
  if policy not in ("R0","R1"):raise ValueError("policy must be R0 or R1")
  self.beta=float(initial_beta);self.policy=policy;self.last_update=-1
 def __call__(self,t,ref_grid,task):
  k=int(round(t/.1))
  if k%5==0 and k!=self.last_update:
   start=int(round(t/.02));window=ref_grid[start:start+N*5:5]
   if len(window)==N:
    f=features(window)
    if not f["low_energy"]:
     lo,hi=(.25,.75) if self.policy=="R0" else (.35,.65)
     raw=float(np.clip((f["rho_high"]-lo)/(hi-lo),0.,1.))
     self.beta=float(np.clip(self.beta+np.clip(raw-self.beta,-.1,.1),0.,1.))
   self.last_update=k
  return self.beta

class StaircaseBetaPolicy:
 """Pre-registered interface diagnostic: 0→.1→.2→.1 at 0.5 s cadence."""
 def __call__(self,t,ref_grid,task):
  n=int(np.floor(t/.5+1e-9));return (0.,.1,.2,.1)[min(n,3)]
