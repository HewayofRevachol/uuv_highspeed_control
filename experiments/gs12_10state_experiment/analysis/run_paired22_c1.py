"""Run the C1/15 m/s 22-state matching slice with four isolated workers."""
import os
os.environ.update({"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"})
import argparse,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from types import SimpleNamespace

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
 a=ap.parse_args();repo=a.repo.resolve();out=a.out.resolve();src=repo/'experiments/gs12_analaysis/frequency_extension_src';sys.path.insert(0,str(src))
 import run_frequency_extension as rf
 ns=SimpleNamespace(repo=repo,out=out,start_index=0,stop_index=None)
 jobs=[(15.,'C1','B1',-1,0.,'primary')]+[(15.,'C1',b,ax,.006,'primary') for b in rf.BANDS for ax in range(3)]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs={pool.submit(rf.execute,ns,*j):i for i,j in enumerate(jobs,1)}
  for f in as_completed(fs):
   i=fs[f];r=f.result();rf.update_manifest(out,r,'main-15-C1-paired',i)
   print(i,r['status'],r['run_id'],r['wall_s_not_realtime_benchmark'],flush=True)
if __name__=='__main__':main()
