"""Execute the frozen GS-12 64-episode categorical PPO pilot."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from .episode_beta import BetaEpisode
from .reward import external_j_track
from .categorical_ppo import BetaActorCritic, update, gae_advantages

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def episode(task, baseline, policy=None, fixed=None, seed=0):
    env=BetaEpisode(task, baseline, initial_beta=.75 if fixed is None else fixed)
    transitions=[]; infos=[]
    while not env.done:
        obs=env.observe()
        if policy is None: action=int(fixed/.25)
        else:
            with torch.no_grad():
                dist=policy.distribution(torch.as_tensor(obs).unsqueeze(0))
                action=int(dist.sample().item())
                logp=float(dist.log_prob(torch.tensor([action])).item())
                value=float(policy.value(torch.as_tensor(obs).unsqueeze(0)).item())
        nxt,reward,done,info=env.step(action); infos.append(info)
        if policy is not None: transitions.append((obs,action,logp,value,reward,done))
    result=env.result(); j=external_j_track(result['arrays']) if result['meta'].get('status') in ('complete_valid','complete_constrained') and env.session.k==640 else None
    return result, infos, transitions, j

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--episodes',type=int,default=64); ap.add_argument('--seed',type=int,default=20261111); ap.add_argument('--out',required=True); a=ap.parse_args()
 out=Path(a.out); (out/'runs').mkdir(parents=True,exist_ok=True); (out/'traces').mkdir(exist_ok=True); (out/'models').mkdir(exist_ok=True)
 torch.manual_seed(a.seed); np.random.seed(a.seed)
 train=rows(out/'tasks_train.jsonl')
 if len(train)!=64 or a.episodes!=64: raise ValueError('protocol requires exactly 64 frozen episodes')
 # Same-task F75 reward baseline cache, persisted incrementally after each complete task.
 cachep=out/'reward_baseline_cache.jsonl'; cache={}
 if cachep.exists(): cache={r['task_id']:r for r in rows(cachep)}
 with cachep.open('a') as f:
  for task in train:
   if task['task_id'] in cache: continue
   start=time.time(); result,info,_,j=episode(task,1e-9,fixed=.75)
   rec={'task_id':task['task_id'],'J_track':j,'status':result['meta'].get('status'),'elapsed_s':time.time()-start}
   f.write(json.dumps(rec)+'\n'); f.flush(); cache[task['task_id']]=rec
   if j is None: raise RuntimeError(f"invalid F75 baseline {task['task_id']}: {rec}")
 agent=BetaActorCritic(); opt=torch.optim.Adam(agent.parameters(),lr=3e-4); log=[]
 for batch in range(4):
  obs=[]; acts=[]; old=[]; rews=[]; vals=[]; dones=[]; task_records=[]
  for task in train[batch*16:(batch+1)*16]:
   result,infos,tr,j=episode(task,cache[task['task_id']]['J_track'],policy=agent)
   task_records.append({'task':task,'result':result,'infos':infos,'transitions':tr,'J':j})
   for o,ac,lp,v,r,d in tr: obs.append(o); acts.append(ac); old.append(lp); vals.append(v); rews.append(r); dones.append(d)
  # Transitions from full tasks; each task's terminal flag ends its own GAE chain.
  obs_t=torch.tensor(np.asarray(obs),dtype=torch.float32); act_t=torch.tensor(acts,dtype=torch.long); old_t=torch.tensor(old); val_t=torch.tensor(vals); rew_t=torch.tensor(rews); done_t=torch.tensor(dones,dtype=torch.bool)
  # Calculate GAE independently per episode to prevent leakage across task resets.
  ret=[]; adv=[]; cursor=0
  for rec in task_records:
   n=len(rec['transitions']); rr=rew_t[cursor:cursor+n]; vv=val_t[cursor:cursor+n]
   nv=torch.cat((vv[1:],torch.zeros(1))); dd=torch.zeros(n,dtype=torch.bool); dd[-1]=True
   aa,rt=gae_advantages(rr,vv,nv,dd,dd,gamma=.98,lam=.95); ret.extend(rt); adv.extend(aa); cursor+=n
  metrics=update(agent,opt,(obs_t,act_t,old_t,torch.stack(ret),torch.stack(adv)),epochs=4,minibatch=64)
  metrics.update(batch=batch+1,episodes=len(task_records),transitions=len(obs),mean_train_J=float(np.mean([x['J'] for x in task_records if x['J'] is not None])))
  log.append(metrics); (out/'train_log.json').write_text(json.dumps(log,indent=2))
  if batch in (1,3):
   torch.save({'agent':agent.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'batch':batch+1,'seed':a.seed},out/f'models/checkpoint_update_{batch+1}.pt')
  for rec in task_records:
   tid=rec['task']['task_id']; rid=f"train_b{batch+1}_{tid}"; rr=rec['result'];
   np.savez_compressed(out/f'traces/{rid}.npz',**rr['arrays'])
   (out/f'runs/{rid}.json').write_text(json.dumps({'task':rec['task'],'status':rr['meta'].get('status'),'J_track':rec['J'],'segments':rec['infos'],'actions':[x[1] for x in rec['transitions']]},default=lambda x:float(x) if isinstance(x,np.floating) else x,indent=2))
 print(json.dumps({'episodes':64,'updates':4,'log':log},indent=2))
if __name__=='__main__': main()
