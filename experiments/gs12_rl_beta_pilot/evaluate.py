"""Paired evaluation on a frozen split; locked split requires selection.json."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np, torch
from .train_ppo import rows,episode
from .episode_beta import BetaEpisode
from .controllers import rule_cap75_action
from .reward import external_j_track
from .categorical_ppo import BetaActorCritic

def run_rule(task):
    env=BetaEpisode(task,1e-9,.75); infos=[]
    while not env.done:
        k=env.session.k; preview=env.session.ref_grid[5*k:5*k+800:5]
        action=rule_cap75_action(preview,env.beta_prev_target)
        _,_,_,info=env.step(action); infos.append(info)
    r=env.result(); j=external_j_track(r['arrays']) if r['meta'].get('status') in ('complete_valid','complete_constrained') and env.session.k==640 else None
    return r,infos,j
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--split',choices=['dev','rl_test_locked_v1'],default='dev');ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out)
 if a.split=='rl_test_locked_v1' and not (out/'selection.json').exists(): raise RuntimeError('locked evaluation requires frozen selection.json')
 tasks=rows(out/f'tasks_{a.split}.jsonl'); methods=[('F00',0.),('F50',.5),('F75',.75),('Rule-cap75',None)]
 (out/'runs').mkdir(exist_ok=True); (out/'traces').mkdir(exist_ok=True)
 models=[]
 for cp in sorted((out/'models').glob('checkpoint_update_*.pt')):
  payload=torch.load(cp,map_location='cpu',weights_only=False); agent=BetaActorCritic();agent.load_state_dict(payload['agent']);agent.eval();models.append((cp.stem,agent,hashlib.sha256(cp.read_bytes()).hexdigest()))
 reports=[]
 for task in tasks:
  for name,beta in methods:
   if beta is None:r,infos,j=run_rule(task)
   else:r,infos,_,j=episode(task,1e-9,fixed=beta)
   rec={'split':a.split,'task_id':task['task_id'],'method':name,'J_track':j,'status':r['meta'].get('status'),'first_violation_s':r['meta'].get('first_violation_s'),'qp_failures':sum(x['qp_failures'] for x in infos),'command_saturation':any(x['command_saturation'] for x in infos),'actual_saturation':any(x['actual_saturation'] for x in infos),'RMSE_E_D_VN':json.dumps(infos[-1]['rmse_E_D_VN']) if infos else None,'peak_E_D_VN':json.dumps(infos[-1]['peak_E_D_VN']) if infos else None,'actions':json.dumps([x['beta_target'] for x in infos])}
   reports.append(rec); rid=f"{a.split}_{task['task_id']}_{name.replace('/','_')}"; np.savez_compressed(out/f'traces/{rid}.npz',**r['arrays']); (out/f'runs/{rid}.json').write_text(json.dumps({'metrics':rec,'segments':infos,'meta':r['meta']} ,indent=2))
  for name,agent,modelsha in models:
   r,infos,tr,j=episode(task,1e-9,policy=agent)
   rec={'split':a.split,'task_id':task['task_id'],'method':name,'checkpoint_sha256':modelsha,'J_track':j,'status':r['meta'].get('status'),'first_violation_s':r['meta'].get('first_violation_s'),'qp_failures':sum(x['qp_failures'] for x in infos),'command_saturation':any(x['command_saturation'] for x in infos),'actual_saturation':any(x['actual_saturation'] for x in infos),'RMSE_E_D_VN':json.dumps(infos[-1]['rmse_E_D_VN']) if infos else None,'peak_E_D_VN':json.dumps(infos[-1]['peak_E_D_VN']) if infos else None,'actions':json.dumps([x['beta_target'] for x in infos])}
   reports.append(rec); rid=f"{a.split}_{task['task_id']}_{name.replace('/','_')}"; np.savez_compressed(out/f'traces/{rid}.npz',**r['arrays']); (out/'runs'/f'{rid}.json').write_text(json.dumps({'metrics':rec,'segments':infos,'meta':r['meta']},indent=2))
  print('completed',task['task_id'],flush=True)
 path=out/f'{a.split}_episode_metrics.csv'
 with path.open('w',newline='') as f:
  fields=list(dict.fromkeys(k for row in reports for k in row)); w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(reports)
 summary={}
 for method in sorted({x['method'] for x in reports}):
  rr=[x for x in reports if x['method']==method]; vals=[x['J_track'] for x in rr if x['J_track'] is not None]
  summary[method]={'complete':len(vals),'tasks':len(rr),'mean_task_J':float(np.mean(vals)) if vals else None,'quality_failures':sum(x['status'] not in ('complete_valid','complete_constrained') for x in rr)}
 (out/f'{a.split}_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
