"""Seven 15 m/s C0 ideal-immediate-actuator diagnostic runs."""
import os
os.environ.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
import argparse,hashlib,json,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

HERE=Path(__file__).resolve().parents[1];SRC=HERE/'src';ANALYSIS=HERE.parent/'gs12_analaysis'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def worker(j):
 repo=Path(j['repo']);sys.path[:0]=[str(repo),str(SRC),str(ANALYSIS/'frequency_extension_src')];os.environ['GS12_REPO']=str(repo)
 import plant_harness
 def instant(self,command):
  command=__import__('numpy').asarray(command).copy()
  self.queue[1:]=self.queue[:-1].copy();self.queue[0]=command
  self.actual=command.copy();self.rate[:]=0.
  self.x=__import__('numpy').asarray(self.step_function(self.x,self.actual,self.parameters)).ravel()
  return self.x.copy()
 plant_harness.Plant.step=instant
 import run_experiment as ten
 return ten._case_run(j)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,default=HERE/'output'/'ideal_diagnostic');ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
 repo=a.repo.resolve();out=a.out.resolve()
 for x in ('runs','traces','matrices'):(out/x).mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(SRC));import run_experiment as ten
 mainp=HERE/'output'/'protocol.json';p=json.loads(mainp.read_text());p['protocol_version']=1;p['diagnostic_plant']='instantaneous actuator; zero command delay; C0 hull';p['main_protocol_sha256']=sha(mainp)
 p['source_sha256']=ten.source_hash()
 proto=out/'protocol.json'
 if proto.exists() and json.loads(proto.read_text())!=p:raise FileExistsError('ideal diagnostic protocol differs')
 if not proto.exists():proto.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')
 jobs=[]
 for band in (None,'B1','B4'):
  if band is None:jobs.append({'speed':15.,'condition':'C0','kind':'straight','base':{'label':'_ideal'},'band':None,'axis':-1,'amp':0.,'phase':'primary','duration':256.})
  else:
   for axis in range(3):jobs.append({'speed':15.,'condition':'C0','kind':'straight','base':{'label':'_ideal'},'band':band,'axis':axis,'amp':.006,'phase':'primary','duration':256.})
 for j in jobs:j.update({'repo':str(repo),'out':str(out)})
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs={pool.submit(worker,j):i for i,j in enumerate(jobs,1)}
  for f in as_completed(fs):r=f.result();print(fs[f],r['status'],r['run_id'],flush=True)
if __name__=='__main__':main()
