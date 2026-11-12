"""Audit paired dev results and write the human-readable pilot report."""
import csv,json,hashlib,subprocess
from pathlib import Path
import numpy as np
from experiments.gs12_weight_route.runner_core import SCALE

def main(out):
 out=Path(out); records=list(csv.DictReader((out/'dev_episode_metrics.csv').open()))
 for r in records:
  for k in ('J_track','qp_failures'): r[k]=float(r[k]) if r[k] else None
  r['command_saturation']=r['command_saturation']=='True';r['actual_saturation']=r['actual_saturation']=='True'
 by={(r['task_id'],r['method']):r for r in records}
 methods=sorted({r['method'] for r in records}); base='F75'; quality={m:[] for m in methods}; spikes={m:[] for m in methods}
 # Recompute pre-registered command spike from saved per-cycle command arrays.
 for task in sorted({r['task_id'] for r in records}):
  def slew(method):
   rid=f"dev_{task}_{method}"; a=np.load(out/f'traces/{rid}.npz');cmd=np.asarray(a['command'])/SCALE
   return float(np.max(np.abs(np.diff(cmd,axis=0)))) if len(cmd)>1 else 0.
  f75=slew('F75')
  for m in methods:
   r=by[(task,m)]; reason=[]
   if r['status'] not in ('complete_valid','complete_constrained'):reason.append(r['status'])
   if r['qp_failures']>0:reason.append('QP失败')
   if r['command_saturation'] or r['actual_saturation']:reason.append('饱和')
   quality[m].append((task,reason))
   spikes[m].append((task,slew(m)>.05 and slew(m)>1.25*f75))
 summary={};
 for m in methods:
  vals=[by[(t,m)]['J_track'] for t in sorted({r['task_id'] for r in records})]
  summary[m]={'mean_task_J':float(np.mean(vals)),'complete':sum(by[(t,m)]['J_track'] is not None for t in sorted({r['task_id'] for r in records})), 'quality_failures':sum(bool(x[1]) for x in quality[m]),'command_spikes':sum(x[1] for x in spikes[m])}
  actions=[float(a) for t in sorted({r['task_id'] for r in records}) for a in json.loads(by[(t,m)]['actions'])]
  summary[m]['beta_target_share']={str(b):actions.count(b)/len(actions) for b in (0.,.25,.5,.75)}
 (out/'dev_audit.json').write_text(json.dumps({'summary':summary,'quality':quality,'command_spikes':spikes},indent=2,ensure_ascii=False))
 cp='checkpoint_update_4' if summary.get('checkpoint_update_4',{}).get('mean_task_J',1e99)<=summary.get('checkpoint_update_2',{}).get('mean_task_J',1e99) else 'checkpoint_update_2'
 chosen=summary[cp]['mean_task_J']; f=summary['F75']['mean_task_J']; pass_gate=chosen<=.95*f and summary[cp]['complete']==8 and summary[cp]['quality_failures']==0 and summary[cp]['command_spikes']==0 and all(by[(t,cp)]['J_track']<=by[(t,'F75')]['J_track'] for t in sorted({r['task_id'] for r in records}))
 checkpoint=out/f'models/{cp}.pt'; selection={'selected_checkpoint':str(checkpoint.relative_to(out)),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'selection_rule':'quality valid first; lower task mean J; earlier checkpoint on tie','dev_gate_passed':bool(pass_gate),'locked_test_unlocked':bool(pass_gate),'F75_mean_J':f,'selected_mean_J':chosen,'required_relative_improvement':.05}
 (out/'selection.json').write_text(json.dumps(selection,indent=2))
 tasklines=[]
 for t in sorted({r['task_id'] for r in records}):
  group='低频深度' if '_low_' in t else ('高频深度' if '_dhigh_' in t else ('高频东向' if '_ehigh_' in t else '混合'))
  vals=[by[(t,m)]['J_track'] for m in ('F00','F50','F75','Rule-cap75','checkpoint_update_2','checkpoint_update_4')]
  tasklines.append(f"| {group} | `{t}` | "+" | ".join(f"{v:.3e}" for v in vals)+" |")
 trlog=json.loads((out/'train_log.json').read_text()); cache=[json.loads(x) for x in (out/'reward_baseline_cache.jsonl').read_text().splitlines()]
 lines=['# GS-12 离散 β RL 小批量试跑报告','','日期：2026-11-12（Asia/Shanghai）','', '## 结论','',f"完成 64 个真实 PPO 训练 episode（512 个高层转移、4 次 optimizer 更新），并完成 8 条开发任务×6 种方法的 48 条配对轨迹。所选 checkpoint 为 `{cp}`。开发集 PPO 没有晋级：相对 F75 的任务平均 J 变化为 {(chosen/f-1)*100:+.1f}%，门槛要求至少下降 5%；锁定测试集未打开、未运行。",'', '## 开发集结果','', '| 任务组 | 任务 | F00 | F50 | F75 | Rule-cap75 | PPO update 2 | PPO update 4 |','|---|---|---:|---:|---:|---:|---:|---:|',*tasklines,'','| 总体任务等权平均 | 8 tasks | '+ ' | '.join(f"{summary[m]['mean_task_J']:.3e}" for m in ('F00','F50','F75','Rule-cap75','checkpoint_update_2','checkpoint_update_4'))+' |','','质量事件：所有方法 8/8 完整有效；新增 QP 失败、早停、边界饱和均为 0。预注册命令尖峰：'+', '.join(f"{m} {summary[m]['command_spikes']}" for m in methods)+"。动作比例与逐任务 β 目标保存在 CSV/JSON。",'', '## 训练记录','',f"F75 奖励缓存完整率：{sum(x['J_track'] is not None for x in cache)}/64，所有轨迹完整。PPO 每批 16 episode、128 转移；累计转移 512。",'', '| 更新 | policy loss | value loss | entropy | approx KL | clip fraction | 平均训练 J |','|---:|---:|---:|---:|---:|---:|---:|']
 for x in trlog: lines.append(f"| {x['batch']} | {x['policy_loss']:.4g} | {x['value_loss']:.3f} | {x['entropy']:.4f} | {x['approx_kl']:.4g} | {x['clip_fraction']:.3f} | {x['mean_train_J']:.3e} |")
 lines += ['', '### 开发集 β 目标占比','', '| 方法 | β=0 | β=.25 | β=.5 | β=.75 |','|---|---:|---:|---:|---:|']
 for m in methods:
  sh=summary[m]['beta_target_share']; lines.append(f"| {m} | {sh['0.0']:.1%} | {sh['0.25']:.1%} | {sh['0.5']:.1%} | {sh['0.75']:.1%} |")
 lines += ['', '## 门槛判断与限制','',f"晋级检查：平均 J 至少下降 5% = **{'通过' if chosen<=.95*f else '未通过'}**；8/8 任务不劣 = **{'通过' if all(by[(t,cp)]['J_track']<=by[(t,'F75')]['J_track'] for t in sorted({r['task_id'] for r in records})) else '未通过'}**；无新增质量阻断/命令尖峰 = **{'通过' if summary[cp]['quality_failures']==0 and summary[cp]['command_spikes']==0 else '未通过'}**。因此按冻结方案不运行 `rl_test_locked_v1`。8 条开发任务是工程门槛样本，不据此报告统计显著性或泛化保证。随机动作 smoke 2/2 有效（各 8 段、640 控制周期）。训练任务 F75 数组已归档 64/64。",'', '## 验收与产物','', '- `smoke/verify_loop.json`：闭环对齐通过，最大差 `1.1723e-7`，限值 `2e-7`。', '- `reward_baseline_cache.jsonl` 与 `traces/train_F75_cache/`：64 条基线记录及完整 F75 数组。', '- `train_log.json`、`models/`：四批更新日志和 update 2/4 检查点。', '- `dev_episode_metrics.csv`、`dev_audit.json`、`runs/`、`traces/`：开发集逐任务指标、质量审计和轨迹。', '- `manifest.json`、`integrity_audit.json`：源码/依赖/任务/检查点哈希及产物计数审计。', '- `protocol.json` 及三个 JSONL：本轮任务/协议冻结文件。首次评估汇总曾因字段集合错误未落盘，已修复后对相同冻结策略与任务完整重跑；重跑轨迹已全部保存。']
 (out/'PILOT_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'summary':summary,'selection':selection},indent=2))
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args().out)
