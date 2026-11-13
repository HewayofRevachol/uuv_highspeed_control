"""Recompute the report tables and figures from saved traces; no simulation or training."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NAMES = ['N_m', 'E_m', 'D_m', 'pitch_rad', 'yaw_rad', 'VN_mps', 'VE_mps', 'VD_mps']
SCALES = np.array([8., 5., 2.])
RUNS = {
    'A_bridge': ['A_bridge_pilot', 'A_bridge_pilot_east', 'A_bridge_remaining'],
    'A_maneuver': ['A_maneuvers'],
    'B_light': ['LB_beta_first', 'LB_beta_hybrid'],
    'C_light': ['LC_speed_first', 'LC_speed_hybrid', 'LC_speed_first_beta0', 'LC_speed_hybrid_beta0'],
    'D_light': ['LD_switch_first', 'LD_switch_hybrid'],
}
EXCLUDED = ['B_static', 'C_half', 'C_range', 'C_mismatch']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (HERE/'tables'/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(error):
    rmse = np.sqrt(np.mean(error**2, axis=0))
    r2 = float(np.mean(np.sum((error[:, [1, 2, 5]]/SCALES)**2, axis=1)))
    return dict(R2=r2, R=float(np.sqrt(r2)), **dict(zip(NAMES, rmse.tolist())))


def masks(task, t):
    result = dict(full=t > 0, steady=t > (80. if task['kind'] == 'bridge' else 16.))
    if task['kind'] == 'switch':
        result.update(stage_1=(t >= 16)&(t < 46), stage_2=(t > 50)&(t < 94),
                      stage_3=t > 98, transition=((t >= 46)&(t <= 50))|((t >= 94)&(t <= 98)))
    return result


def eligible(row):
    return row['R2'] is not None and not any(row['command_saturation']+row['actual_saturation'])


def savefig(fig, name):
    for suffix in ['png', 'pdf']:
        fig.savefig(HERE/'figures'/f'{name}.{suffix}', dpi=190, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'outputs/gs20_continuation')
    args = parser.parse_args()
    data = args.data_root
    font_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(font_path)
    font_family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family': font_family, 'font.size': 10,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    rows, file_manifest, flat, window_rows = [], [], [], []
    errors, references = {}, {}
    for stage, folders in RUNS.items():
        for folder in folders:
            directory = data/folder
            protocol = json.loads((directory/'protocol.json').read_text())
            file_manifest.extend([dict(path=f'{folder}/{n}', sha256=sha(directory/n))
                                  for n in ['protocol.json', 'episodes.jsonl']])
            for line in (directory/'episodes.jsonl').read_text().splitlines():
                row = json.loads(line)
                row.update(stage=stage, directory=folder)
                task = row['task']
                trace = directory/row['trace']
                file_manifest.append(dict(path=f'{folder}/{row["trace"]}', sha256=sha(trace)))
                with np.load(trace, allow_pickle=False) as a:
                    t, e = a['time'], a['error']
                    assert len(t) == len(e) and np.all(np.isfinite(e))
                    assert int(np.sum(~a['qp_ok'])) == row['qp_failures']
                    np.testing.assert_allclose(t[-1], row['observed_end_s'], atol=1e-12)
                    np.testing.assert_allclose(np.diff(t), .02, rtol=1e-9, atol=1e-12)
                    for window, mask in masks(task, t).items():
                        old = row['scores'][window]
                        if not np.any(mask):
                            assert old is None
                            continue
                        s = score(e[mask])
                        np.testing.assert_allclose(s['R2'], old['R2'], rtol=1e-11, atol=1e-20)
                        np.testing.assert_allclose([s[n] for n in NAMES],
                                                   [old['rmse'][n] for n in NAMES], rtol=1e-11, atol=1e-14)
                        window_rows.append(dict(stage=stage, task_id=task['task_id'], method=row['method'],
                                                window=window, complete=row['complete'], samples=int(mask.sum()), **s))
                    s = score(e)
                    if row['R2'] is not None:
                        np.testing.assert_allclose(s['R2'], row['R2'], rtol=1e-11)
                    errors[stage, task['task_id'], row['method']] = (t.copy(), e.copy())
                    if stage == 'A_bridge':
                        references[task['plant'], task['axis'], row['method']] = a['reference'].copy()
                    if stage == 'D_light':
                        row['_beta'] = a['beta'].copy()
                flat.append(dict(stage=stage, task_id=task['task_id'], method=row['method'],
                    plant=task['plant'], axis=task.get('axis', task.get('maneuver')),
                    speed_mps=task['speed_mps'], duration_s=task['duration_s'], k=task.get('k'),
                    order=task.get('order'), phase_rad=task.get('phase_rad'), amplitude=task['amplitude'],
                    complete=row['complete'], eligible=eligible(row), qp_failures=row['qp_failures'],
                    observed_end_s=row['observed_end_s'], R2_valid=row['R2'], R2_observed=s['R2'],
                    R_observed=s['R'], **{n:s[n] for n in NAMES},
                    command_saturation_any=any(row['command_saturation']),
                    actual_saturation_any=any(row['actual_saturation']),
                    wall_s=row['wall_s'], source=f'{folder}/{row["trace"]}', initial=protocol['initial']))
                rows.append(row)
    keys = [(r['stage'], r['task']['task_id'], r['method']) for r in rows]
    assert len(set(keys)) == len(keys) == 260
    expected = {'A_bridge':24, 'A_maneuver':108, 'B_light':96, 'C_light':16, 'D_light':16}
    assert all(sum(r['stage'] == stage for r in rows) == count for stage, count in expected.items())
    assert all(eligible(r) for r in rows if r['stage'] in ['B_light','C_light','D_light'])
    write_csv('all_runs.csv', flat)
    write_csv('window_scores.csv', window_rows)

    # Reconstruct both 64-second FRF windows, including paired zero subtraction.
    omega = 2*np.pi*np.array([2,4,6,8,12,16,24])/64
    frfs, frf_rows, bridge_checks = {}, [], []
    for plant in ['combined','matched']:
        for method in ['first:0','first:1','hybrid:1']:
            group = {r['task']['axis']:r for r in rows if r['stage']=='A_bridge'
                     and r['task']['plant']==plant and r['method']==method}
            baseline = errors['A_bridge', group['zero']['task']['task_id'], method][1][4::5][:,[1,2,5]]/SCALES
            H = np.empty((2,7,3,3),complex)
            for axis,name in enumerate(['east','depth','speed']):
                t,e = errors['A_bridge',group[name]['task']['task_id'],method]
                t=t[4::5]
                e=e[4::5][:,[1,2,5]]/SCALES-baseline
                ref=references[plant,name,method][4::5]
                inp=np.column_stack((ref[:,1],ref[:,2]-20,ref[:,5]-15))/SCALES
                for wi,(lo,hi) in enumerate([(16.,80.),(80.,144.)]):
                    mask=(t>lo+1e-9)&(t<=hi+1e-9)
                    assert mask.sum()==640
                    proj=2/640*np.exp(-1j*omega[:,None]*t[mask])
                    H[wi,:,:,axis]=(proj@e[mask])/(proj@inp[mask,axis])[:,None]
            saved=data/'A_frf_full_v2'/f'{method.replace(":","_")}__{plant}.npz'
            with np.load(saved) as z:
                np.testing.assert_allclose(H[1],z['H'],rtol=1e-10,atol=1e-13)
                np.testing.assert_allclose(H[0],z['H_period1'],rtol=1e-10,atol=1e-13)
            sv=np.linalg.svd(H[1],compute_uv=False)
            frfs[plant,method]=sv[:,0]
            delta=np.linalg.norm(H[1]-H[0],axis=(1,2))/np.linalg.norm(H[1],axis=(1,2))
            for k,w,s,d in zip([2,4,6,8,12,16,24],omega,sv,delta):
                frf_rows.append(dict(plant=plant,method=method,k=k,omega_radps=w,
                                     sigma1=s[0],sigma2=s[1],sigma3=s[2],period_relative_change=d))
            if method=='first:0':
                archive=ROOT/'experiments/gs12_analaysis/GS-12_frequency_data.zip'
                with zipfile.ZipFile(archive) as z:
                    with np.load(io.BytesIO(z.read(f'gs12_followup/frequency_data/frf_v15_{plant}_a0.006.npz'))) as old:
                        diff=np.linalg.norm(H[1]-old['H'],axis=(1,2))/np.linalg.norm(old['H'],axis=(1,2))
                assert np.all(diff<1e-3)
                bridge_checks.append(dict(plant=plant,max_relative_matrix_difference=float(max(diff)),passed=True))
    write_csv('A_frequency.csv',frf_rows)

    # Per-method summary keeps the denominator and failures visible.
    summary=[]
    for stage in RUNS:
        selected=[r for r in rows if r['stage']==stage]
        for method in sorted({r['method'] for r in selected}):
            rs=[r for r in selected if r['method']==method]
            valid=[r for r in rs if r['R2'] is not None]
            summary.append(dict(stage=stage,method=method,attempted=len(rs),complete=sum(r['complete'] for r in rs),
                eligible=sum(eligible(r) for r in rs),qp_failures=sum(r['qp_failures'] for r in rs),
                mean_R2_complete_only=float(np.mean([r['R2'] for r in valid])) if valid else None))
    write_csv('stage_summary.csv',summary)

    # A common-prefix table is diagnostic only; it never ranks full-path failures.
    common=[]
    for task_id in sorted({r['task']['task_id'] for r in rows if r['stage']=='A_maneuver'}):
        rs=[r for r in rows if r['stage']=='A_maneuver' and r['task']['task_id']==task_id]
        end=min(r['observed_end_s'] for r in rs)
        for r in rs:
            t,e=errors['A_maneuver',task_id,r['method']]
            common.append(dict(task_id=task_id,method=r['method'],common_end_s=end,
                               complete=r['complete'],**score(e[t<=end+1e-9])))
    write_csv('A_maneuver_common_prefix.csv',common)

    selection=json.loads((data/'LB_selection.json').read_text())
    b_rows=[]
    for task_id in sorted({r['task']['task_id'] for r in rows if r['stage']=='B_light'}):
        for structure in ['first','hybrid']:
            rs=[r for r in rows if r['stage']=='B_light' and r['task']['task_id']==task_id
                and r['structure']==structure]
            best=min(rs,key=lambda r:r['R2'])
            t=best['task']
            out=dict(task_id=task_id,structure=structure,plant=t['plant'],axis=t['axis'],k=t['k'],
                     best_beta=float(best['method'].split(':')[1]))
            for r in rs:
                beta=r['method'].split(':')[1]
                out[f'R2_beta_{beta}']=r['R2']
                for name in ['E_m','D_m','VN_mps']:
                    out[f'{name}_beta_{beta}']=r['scores']['full']['rmse'][name]
            b_rows.append(out)
    write_csv('B_per_task_beta.csv',b_rows)

    pairs=[]
    for stage in ['C_light','D_light']:
        for structure in ['first','hybrid']:
            rs=[r for r in rows if r['stage']==stage and r['structure']==structure]
            for tid in sorted({r['task']['task_id'] for r in rs}):
                task_rows=[r for r in rs if r['task']['task_id']==tid]
                base=next(r for r in task_rows if r['method']==f'{structure}:' + ('0' if stage=='C_light' else 'best'))
                cand=next(r for r in task_rows if r is not base)
                t=base['task']
                p=dict(stage=stage,structure=structure,task_id=tid,axis=t['axis'],order=t.get('order'),
                       speed_mps=t['speed_mps'],baseline=base['method'],candidate=cand['method'],
                       baseline_R2=base['R2'],candidate_R2=cand['R2'],
                       R2_change_pct=100*(cand['R2']/base['R2']-1),
                       win=cand['R2']<base['R2'],tie=cand['R2']==base['R2'])
                for name in ['E_m','D_m','VN_mps']:
                    p[f'baseline_{name}']=base['scores']['full']['rmse'][name]
                    p[f'candidate_{name}']=cand['scores']['full']['rmse'][name]
                pairs.append(p)
    write_csv('C_D_pairs.csv',pairs)
    for structure in ['first','hybrid']:
        for stage in ['B_light','C_light','D_light']:
            stage_rows=[r for r in rows if r['stage']==stage and r['structure']==structure]
            if stage=='B_light':
                means={b:float(np.mean([r['R2'] for r in stage_rows if r['method']==f'{structure}:{b:g}'])) for b in [0,.5,1]}
                for b,value in means.items():
                    np.testing.assert_allclose(value,selection[structure]['fixed_grid_mean_R2'][str(float(b))],rtol=1e-12)
                assert min(means,key=means.get)==selection[structure]['fixed_beta']==.5
            else:
                ps=[p for p in pairs if p['stage']==stage and p['structure']==structure]
                aggregate=1-np.mean([p['candidate_R2'] for p in ps])/np.mean([p['baseline_R2'] for p in ps])
                saved=json.loads((data/('LC_analysis_paired' if stage=='C_light' else 'LD_analysis')/'summary.json').read_text())
                if stage=='D_light':
                    np.testing.assert_allclose(aggregate,saved['paired'][f'{structure}:rule']['mean_R2_reduction_fraction'],rtol=1e-12)
                    assert sum(p['win'] for p in ps)==0 and sum(p['tie'] for p in ps)==2

    # Three compact figures: FRF conditions, beta preference, and transfer/scheduling results.
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.0),layout='constrained')
    for ax,plant,title in zip(axs,['combined','matched'],['组合失配对象','匹配对象']):
        for method,label,color in [('first:0','原22状态','#2463a6'),('first:1','保守22状态','#dd7b24'),('hybrid:1','混合结构','#318a67')]:
            ax.semilogy(omega,frfs[plant,method],marker='o',label=label,color=color)
        ax.set(title=title,xlabel='频率 ω（rad/s）',ylabel='最大奇异值 σmax（无量纲）')
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    savefig(fig,'01_frequency')

    fig,axs=plt.subplots(1,2,figsize=(11.2,7.5),layout='constrained')
    labels=[]
    for structure,ax in zip(['first','hybrid'],axs):
        rs=[r for r in b_rows if r['structure']==structure]
        values=np.array([[r[f'R2_beta_{b}']/r['R2_beta_0'] for b in ['0','0.5','1']] for r in rs])
        logs=np.log10(values)
        im=ax.imshow(logs,cmap='RdBu_r',norm=TwoSlopeNorm(vcenter=0,vmin=-1.5,vmax=1.5),aspect='auto')
        labels=[f'{"匹配" if r["plant"]=="matched" else "失配"} · {dict(east="东向",depth="深度",speed="速度",coupled="耦合")[r["axis"]]} · k={r["k"]}' for r in rs]
        ax.set_yticks(range(16),labels,fontsize=8.5)
        ax.set_xticks(range(3),['β=0','β=0.5','β=1']); ax.set_title(structure+'：相对β=0的全程R²倍率')
        for i in range(16):
            for j in range(3):
                ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(logs[i,j])>1 else 'black')
    fig.colorbar(im,ax=axs,label='log10(R² / R²β=0)，蓝色表示更低误差',shrink=.7)
    savefig(fig,'02_beta_grid')

    fig,axs=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
    cp=[p for p in pairs if p['stage']=='C_light' and p['structure']=='first']
    x=np.arange(4)
    base=[p['baseline_D_m']*1000 for p in cp]; cand=[p['candidate_D_m']*1000 for p in cp]
    axs[0].bar(x-.18,base,width=.36,label='β=0',color='#2463a6')
    axs[0].bar(x+.18,cand,width=.36,label='β=0.5',color='#318a67')
    axs[0].set_xticks(x,[f'{int(p["speed_mps"])} m/s\n'+('深度' if p['axis']=='depth' else '耦合') for p in cp])
    axs[0].set(title='C：first结构的深度RMSE',ylabel='全程深度RMSE（毫米）'); axs[0].legend()
    for offset,structure,color in [(-.18,'first','#dd7b24'),(.18,'hybrid','#8d6ba6')]:
        dp=[p for p in pairs if p['stage']=='D_light' and p['structure']==structure]
        axs[1].bar(x+offset,[p['R2_change_pct'] for p in dp],width=.36,label=structure,color=color)
    axs[1].set_xticks(x,['东向\n低高低','东向\n高低高','深度\n低高低','深度\n高低高'])
    axs[1].set(title='D：规则相对固定β=0.5',ylabel='全程R²变化（%，正值表示变差）'); axs[1].legend()
    for ax in axs: ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    savefig(fig,'03_transfer_schedule')

    # Source provenance and bounded audit coverage. No re-run of closed-loop experiments.
    excluded=[]
    for name in EXCLUDED:
        p=data/name/'episodes.jsonl'
        excluded.append(dict(directory=name,records=len(p.read_text().splitlines()),reason='original full-grid attempt incomplete; excluded from reported groups',sha256=sha(p)))
    manifest=dict(data_root_relative_to_workspace='outputs/gs20_continuation',
                  included_records=len(rows),files=file_manifest,excluded_groups=excluded,
                  source_snapshot_files=[dict(path=str(p.relative_to(HERE)),sha256=sha(p))
                                         for p in sorted((HERE/'source_snapshot').rglob('*')) if p.is_file()])
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    audit=dict(verification_status='ANALYZED',new_closed_loop_simulations=0,new_training_runs=0,
        included_records=len(rows),stage_records=expected,
        raw_trace_rmse_and_R2='260/260 matched stored values; all available windows checked',
        score_tolerance=dict(rtol=1e-11,rmse_atol=1e-14),
        saved_FRF_matrices='6/6 matched both windows; rtol=1e-10, atol=1e-13',
        legacy_bridge_checks=bridge_checks,light_records_all_eligible=True,
        D_gate=dict(first=dict(wins=0,ties=2,losses=2,required_wins=3,triggered=False),
                    hybrid=dict(wins=0,ties=2,losses=2,required_wins=3,triggered=False)),
        excluded_records=sum(x['records'] for x in excluded),
        limitations=['No fresh closed-loop rerun; verifies saved data and arithmetic only.',
                     'No live noise/current/hardware experiments; no confidence intervals or statistical significance.',
                     'Reported saturation flags follow original instrumentation; constraint implementation was not independently certified.'])
    (HERE/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
