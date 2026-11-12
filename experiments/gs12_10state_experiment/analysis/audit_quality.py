"""Audit exact W2/W3 saturation and precheck gates from run traces."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

SCALE=np.array([1500000.,.7853981633974483,.7853981633974483]);LOW=np.array([0.,-1.,-1.])
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def saturated(d,lo,hi):
 c=d['command'][ (d['time_control_s']>lo)&(d['time_control_s']<=hi)]/SCALE
 a=d['actual_substep'][(d['time_substep_s']>lo)&(d['time_substep_s']<=hi)]/SCALE
 if not len(c) or not len(a):return None
 return {'command':bool(np.any((c>=1-1e-6)|(c<=LOW+1e-6))),
         'actual':bool(np.any((a>=1-1e-6)|(a<=LOW+1e-6)))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;ph=digest(out/'protocol.json');runs=[]
 for p in (out/'runs').glob('*.json'):
  r=json.loads(p.read_text())
  if r.get('protocol_sha256')!=ph or r.get('phase_id') not in ('primary','secondary'):continue
  d=np.load(out/'traces'/(r['run_id']+'.npz'))
  r['window_saturation']={f'W{i}':saturated(d,*rng) for i,rng in ((2,(128,192)),(3,(192,256)))}
  r['passes_no_saturation_windows']=all(not any(v.values()) for v in r['window_saturation'].values() if v)
  runs.append(r)
 grouped={}
 for r in runs:
  key=(r['speed_mps'],r['condition'],r['operating_regime'],r.get('regime_parameters',{}).get('label',''))
  grouped.setdefault(key,[]).append(r)
 report=[]
 for key,rs in sorted(grouped.items(),key=lambda x:str(x[0])):
  zero=next((r for r in rs if r['axis']<0),None)
  ex=[r for r in rs if r['axis']>=0]
  by={}
  for r in ex:
   k=r.get('probe_k') if r['operating_regime'] in ('s_bend','coupled') else r.get('band')
   by.setdefault(str(k),[]).append(r)
  groups=[]
  for k,rr in sorted(by.items()):
   axes={str(r['axis']):{'status':r['status'],'observed_s':r['observed_duration_s'],'no_sat_W2_W3':r['passes_no_saturation_windows'],'sat':r['window_saturation']} for r in rr}
   complete_axes=len(rr)==3 and {r['axis'] for r in rr}=={0,1,2}
   full=complete_axes and all(r['status']=='complete_valid' and r['passes_no_saturation_windows'] for r in rr)
   groups.append({'probe_or_band':k,'axes':axes,'complete_full_quality_group':full})
  report.append({'speed_mps':key[0],'condition':key[1],'regime':key[2],'label':key[3],
   'zero':None if zero is None else {'status':zero['status'],'duration_s':zero['observed_duration_s'],'no_sat_W2_W3':zero['passes_no_saturation_windows'],'sat':zero['window_saturation']},
   'excitation_groups':groups})
 dst=out/'quality_gate_audit.json';dst.write_text(json.dumps({'protocol_sha256':ph,'runs_audited':len(runs),'groups':report},indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'runs_audited':len(runs),'groups':len(report),'output':str(dst)},indent=2))
if __name__=='__main__':main()
