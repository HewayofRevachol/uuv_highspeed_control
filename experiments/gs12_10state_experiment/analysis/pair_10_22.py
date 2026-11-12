"""Exact-reference C1/15 m/s 10-state versus 22-state FRF pairing."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dft(x,k):return 2/len(x)*np.exp(-2j*np.pi*np.arange(len(x))*k/len(x))@x
def pack(z):return {'real':np.asarray(z).real.tolist(),'imag':np.asarray(z).imag.tolist()}
def runs(path,protocol=None):
 out=[]
 for p in (path/'runs').glob('*.json'):
  r=json.loads(p.read_text())
  if protocol is not None and r.get('protocol_sha256')!=protocol:continue
  out.append(r)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
 p10=out/'protocol.json';p22=out/'paired22_exact'/'protocol.json';h10=sha(p10)
 r10=runs(out,h10);r22=runs(out/'paired22_exact')
 def index(rs):return {(r['speed_mps'],r['condition'],r.get('band'),r.get('axis'),r.get('amplitude_per_line'),r.get('phase_id')):r for r in rs}
 i10=index([r for r in r10 if r.get('operating_regime')=='straight' and r.get('speed_mps')==15. and r.get('condition')=='C1' and r.get('phase_id')=='primary'])
 i22=index([r for r in r22 if r.get('speed_mps')==15. and r.get('condition')=='C1' and r.get('phase_id')=='primary'])
 paired=[];missing=[]
 for band,bins in ((f'B{i+1}',list(range(8*i+1,8*i+9))) for i in range(4)):
  axes10=[i10.get((15.,'C1',band,ax,.006,'primary')) for ax in range(3)]
  axes22=[i22.get((15.,'C1',band,ax,.006,'primary')) for ax in range(3)]
  z10=i10.get((15.,'C1',None,-1,0.,'primary'));z22=i22.get((15.,'C1',None,-1,0.,'primary'))
  if not z10 or not z22 or not all(axes10) or not all(axes22):missing.append({'band':band,'reason':'axis or common zero record missing'});continue
  allr=axes10+[z10]+axes22+[z22]
  if any(r.get('status') not in ('complete_valid','complete_constrained') or abs(r.get('observed_duration_s',0)-256)>1e-8 for r in allr):missing.append({'band':band,'reason':'incomplete/failed trace'});continue
  def load(base,r):return np.load(base/'traces'/(r['run_id']+'.npz'))
  d10=[load(out,r) for r in axes10];d22=[load(out/'paired22_exact',r) for r in axes22];b10=load(out,z10);b22=load(out/'paired22_exact',z22)
  refdiff=[];reflen=[];plantok=[]
  for x,y,rx,ry in zip(d10,d22,axes10,axes22):
   a=x['reference_substep'];b=y['reference_substep'];refdiff.append(float(np.max(np.abs(a-b))));reflen.append(len(a));plantok.append(rx['plant_config']==ry['plant_config'])
  if not all(plantok) or max(refdiff)>1e-10:
   missing.append({'band':band,'reason':'physical plant or reference arrays differ','reference_max_abs':max(refdiff),'plant_config_match':all(plantok)});continue
  windows={}
  for name,lo,hi in (('W1',64,128),('W2',128,192),('W3',192,256)):
   H10=np.zeros((8,3,3),complex);H22=np.zeros_like(H10)
   for ax,(x,y) in enumerate(zip(d10,d22)):
    for ds,b,zout,key in ((x,b10,H10,'error_path_normalized'),(y,b22,H22,'error_normalized')):
     t=ds['time_control_s'];m=(t>lo+1e-9)&(t<=hi+1e-9)
     err=ds[key][m]-b['error_path_normalized' if key=='error_path_normalized' else 'error_normalized'][m]
     u=ds['input_normalized'][m,ax]
     for j,k in enumerate(bins):zout[j,:,ax]=dft(err,k)/dft(u,k)
   diff=H10-H22;sv10=np.linalg.svd(H10,compute_uv=False);sv22=np.linalg.svd(H22,compute_uv=False)
   windows[name]={'H10':pack(H10),'H22':pack(H22),'H10_minus_H22':pack(diff),
    'relative_frobenius_by_bin':(np.linalg.norm(diff,axis=(1,2))/np.maximum(np.linalg.norm(H22,axis=(1,2)),1e-8)).tolist(),
    'absolute_frobenius_by_bin':np.linalg.norm(diff,axis=(1,2)).tolist(),
    'singular_values_10':sv10.tolist(),'singular_values_22':sv22.tolist(),
    'sigma1_difference':(sv10[:,0]-sv22[:,0]).tolist(),
    'column_norms_10':np.linalg.norm(H10,axis=1).tolist(),'column_norms_22':np.linalg.norm(H22,axis=1).tolist()}
  paired.append({'speed_mps':15.,'plant_condition':'C1','band':band,'bins':bins,'omega_radps':(2*np.pi*np.array(bins)/64).tolist(),
   'reference_max_abs_10_vs_22':max(refdiff),'reference_samples_per_axis':reflen,'plant_config_exact_match':all(plantok),
   'runs_10':[r['run_id'] for r in axes10],'runs_22':[r['run_id'] for r in axes22],'zero_10':z10['run_id'],'zero_22':z22['run_id'],'windows':windows})
 result={'protocol_10_sha256':h10,'protocol_22_sha256':sha(p22),'analysis_source_sha256':sha(Path(__file__)),
  'pairing_rule':'C1, 15 m/s, same PlantConfig and elementwise identical reference_substep array; both use primary phase and a0.006.',
  'paired_groups':len(paired),'missing':missing,'pairs':paired,
  'interpretation':'Empirical closed-loop reference-to-error comparison; this C1 slice does not establish general 10/22-state superiority.'}
 (out/'matrices'/'paired_10_22_C1_15.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'paired_groups':len(paired),'missing':missing,'max_reference_difference':max((x['reference_max_abs_10_vs_22'] for x in paired),default=None)},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
