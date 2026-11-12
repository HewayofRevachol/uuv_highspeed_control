"""Run 22-state C1/15 m/s records using the frozen 10-state reference arrays."""
import os
os.environ.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
import argparse,hashlib,json,importlib.metadata,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1];SRC=HERE/'src';ANALYSIS=HERE.parent/'gs12_analaysis';OUTDEFAULT=HERE/'output'/'paired22_exact'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def jwrite(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def source_hash():return {p.name:sha(p) for p in SRC.glob('*.py')}
def protocol(repo,out):
 names=['simulator/batch_mpc.py','simulator/model.py','simulator/rti_symbolic.py','aav_mpc/condensed_mpc.py',
        'experiments/gs12_analaysis/frequency_extension_src/plant_harness.py']
 versions={}
 for n in ('numpy','torch','casadi','osqp','scipy'):
  try:versions[n]=importlib.metadata.version(n)
  except importlib.metadata.PackageNotFoundError:versions[n]=None
 p10=out.parent/'protocol.json' if out.name=='paired22_exact' else HERE/'output'/'protocol.json'
 return {'experiment':'GS-12 exact-reference 22-state C1 slice','controller_state_dim':22,'paired_10_protocol_sha256':sha(p10),
  'source_sha256':source_hash(),'plant_harness_sha256':{n:sha(repo/n) for n in names},
  'reference_source_sha256':sha(SRC/'reference10.py'),'packages':versions,'speed_mps':15.,'condition':'C1',
  'bands':{f'B{i+1}':list(range(8*i+1,8*i+9)) for i in range(4)},'amplitude_per_line':.006,'phase':'primary',
  'duration_s':256.,'dt_s':.1,'plant_substep_s':.02,'purpose':'paired comparator; same reference generator and arrays as 10-state records'}
def run_case(j):
 repo=Path(j['repo']);out=Path(j['out']);sys.path[:0]=[str(repo),str(SRC),str(ANALYSIS/'frequency_extension_src')]
 os.environ['GS12_REPO']=str(repo)
 import torch;torch.set_num_threads(1)
 from plant_harness import PlantConfig,run_closed_loop,SCALE
 from reference10 import straight
 import run_experiment as ten
 band,axis=j['band'],j['axis'];phase='primary';amp=.006 if axis>=0 else 0.
 ref=straight(15.,band if axis>=0 else None,axis,amp,phase)
 psha=sha(out/'protocol.json');tag=band if band else 'zero'
 rid=f'p{psha[:12]}_model22_v15_C1_straight_{tag}_axis{axis}_a{amp:g}_{phase}'
 jp=out/'runs'/(rid+'.json');tp=out/'traces'/(rid+'.npz')
 if jp.exists() or tp.exists():
  if not(jp.exists() and tp.exists()):raise FileExistsError(f'partial record {rid}')
  old=json.loads(jp.read_text())
  if old.get('protocol_sha256')!=psha or old.get('source_sha256')!=source_hash():raise FileExistsError(f'stale record {rid}')
  return old
 cfg=PlantConfig(elevator=.9,rudder=.9)
 t0=time.perf_counter();result=run_closed_loop(cfg,15.,ref,256.,1.,1.,None,True);meta,ar=result['meta'],result['arrays']
 idx=np.arange(4,len(ar['time']),5);tc=ar['time'][idx]
 if not np.allclose(tc,np.arange(1,len(tc)+1)*.1,rtol=0,atol=1e-10):raise ValueError('control sample alignment failed')
 trace={'time_substep_s':ar['time'],'state_substep':ar['state'],'reference_substep':ar['reference'],'output_substep':ar['output'],
  'error_substep':ar['error'],'diagnostic_substep':ar['diagnostic'],'actual_substep':ar['actual'],'queue_substep':ar['queue'],
  'command_substep':ar['subcommand'],'time_control_s':tc,'reference_control':ar['reference'][idx],
  'output_control':ar['output'][idx], 'error_normalized':ar['error'][idx][:,[1,2,5]]/SCALE,
  'error_path_normalized':ar['error'][idx][:,[1,2,5]]/SCALE,
  'input_normalized':np.column_stack((ar['reference'][idx,1]/8.,(ar['reference'][idx,2]-20.)/5.,(ar['reference'][idx,5]-15.)/2.)),
  'command':ar['command'],'qp_ok':ar['qp_ok'],'qp_primal':ar['qp_primal'],'qp_dual':ar['qp_dual'],
  'one_step_error':ar['one_step_error'],'omega_radps':2*np.pi*np.arange(1,33)/64}
 tmp=tp.with_suffix('.npz.tmp');
 with open(tmp,'wb') as f:np.savez_compressed(f,**trace)
 tmp.replace(tp)
 ref_sha=sha_bytes(np.asarray(ref(np.arange(0.,256.+35*.1+1e-9,.02)),dtype='<f8').tobytes())
 status='early_violation' if meta['first_diagnostic_limit_violation'] else 'qp_failure' if meta['qp_failures'] else 'complete_constrained' if any(v>0 for v in meta['command_saturation_fraction']+meta['actual_saturation_fraction_substeps']) else 'complete_valid'
 rec={'run_id':rid,'protocol_sha256':psha,'paired_10_protocol_sha256':sha(HERE/'output'/'protocol.json'),
  'source_sha256':source_hash(),'reference_sha256':ref_sha,'controller_state_dim':22,'controller':'CondensedActuatorMPC',
  'speed_mps':15.,'condition':'C1','plant_config':meta['config'],'band':band,'axis':axis,'amplitude_per_line':amp,'phase_id':phase,
  'status':status,'observed_duration_s':meta['duration_observed_s'],'control_periods':len(ar['command']),'substeps':len(ar['time']),
  'first_violation':meta['first_diagnostic_limit_violation'],'qp_failures':meta['qp_failures'],
  'command_saturation_fraction':meta['command_saturation_fraction'],'actual_saturation_fraction':meta['actual_saturation_fraction_substeps'],
  'trace_sha256':sha(tp),'wall_s':time.perf_counter()-t0}
 jwrite(jp,rec);return rec
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,default=OUTDEFAULT);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();repo=a.repo.resolve();out=a.out.resolve()
 for d in ('runs','traces','matrices'): (out/d).mkdir(parents=True,exist_ok=True)
 p=protocol(repo,out);pp=out/'protocol.json'
 if pp.exists() and json.loads(pp.read_text())!=p:raise FileExistsError('22 exact-reference protocol differs; choose a fresh output directory')
 if not pp.exists():jwrite(pp,p)
 jobs=[{'band':None,'axis':-1}]+[{'band':f'B{i+1}','axis':ax} for i in range(4) for ax in range(3)]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs={pool.submit(run_case,{**j,'repo':str(repo),'out':str(out)}):i for i,j in enumerate(jobs,1)}
  for f in as_completed(fs):r=f.result();print(fs[f],r['status'],r['run_id'],r['wall_s'],flush=True)
if __name__=='__main__':main()
