"""Archive full F75 trajectories for the same frozen training tasks."""
import json,time
from pathlib import Path
import numpy as np
from .episode_beta import BetaEpisode

def main(out):
 out=Path(out); tasks=[json.loads(x) for x in (out/'tasks_train.jsonl').read_text().splitlines()]
 cache={r['task_id']:r for r in (json.loads(x) for x in (out/'reward_baseline_cache.jsonl').read_text().splitlines())}
 d=out/'traces'/'train_F75_cache';d.mkdir(exist_ok=True)
 for i,task in enumerate(tasks):
  path=d/f"{task['task_id']}.npz"
  if path.exists():continue
  env=BetaEpisode(task,cache[task['task_id']]['J_track'],.75)
  while not env.done:env.step(3)
  result=env.result()
  np.savez_compressed(path,**result['arrays'])
  assert env.session.k==640 and result['meta']['status'] in ('complete_valid','complete_constrained')
  print(i+1,task['task_id'],flush=True)
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
