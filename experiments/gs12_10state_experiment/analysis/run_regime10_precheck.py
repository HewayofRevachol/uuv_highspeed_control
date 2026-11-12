"""Run the pre-registered 10 m/s turn, slope and periodic baseline gates."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json,sys

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
 a=ap.parse_args();src=Path(__file__).resolve().parents[1]/'src';sys.path.insert(0,str(src));import run_experiment as r
 jobs=r._stage_jobs('regime-precheck')
 for j in jobs:j['speed']=10.;j['repo']=str(a.repo.resolve());j['out']=str(a.out.resolve())
 p=a.out/'precheck_10m_progress.json';done=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(r._case_run,j):i for i,j in enumerate(jobs,1)}
  for f in as_completed(futures):
   i=futures[f];rec=f.result();done.append(rec['run_id']);r._manifest(a.out,rec,'regime-precheck-10m',i,len(jobs))
   p.write_text(json.dumps({'stage':'regime-precheck-10m','completed':len(done),'planned':len(jobs),'last_run_id':rec['run_id'],'last_status':rec['status']},indent=2)+'\n')
   print(i,rec['status'],rec['run_id'],flush=True)
 print(json.dumps({'completed':len(done),'planned':len(jobs)},indent=2))
if __name__=='__main__':main()
