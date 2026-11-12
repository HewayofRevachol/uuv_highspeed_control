"""Freeze the GS-12 RL pilot task lists once; existing files are never replaced."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def digest(data): return hashlib.sha256(data).hexdigest()
def canonical(x): return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def base(task_id, split, kind, **kw):
    return dict(task_id=task_id, split=split, kind=kind, duration_s=64., amplitude=.006,
                speed_mps=15., window_s=[32.,64.], **kw)

def freeze(out):
    out=Path(out); out.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(20261111); train=[]
    centers=((19,39),(21,43),(25,45),(27,47))
    for batch in range(4):
      for kind,axis,ks in (("low_depth","depth",(1,3,4,6)),("high_depth","depth",(25,27,29,31)),("high_east","east",(25,27,29,31))):
       for k in ks:
        train.append(base(f"train_{kind}_{k}_{batch}","train", "single", axis=axis,k=k,phase_rad=round(float(rng.uniform(-np.pi,np.pi)),6),batch=batch))
      for j,c in enumerate(centers):
       train.append(base(f"train_mixed_{batch}_{j}","train","mixed",axis="east_depth",switch_times_s=list(c),phase_rad=round(float(rng.uniform(-np.pi,np.pi)),6),batch=batch))
    # Four classes, sixteen tasks each, ordered in four balanced batches.
    assert len(train)==64
    dev=[]
    for k,p in ((2,.42),(8,-1.05),(26,.74),(30,-1.27)): dev.append(base(f"rl_dev_{'low' if k<10 else 'dhigh'}_{k}","dev","single",axis="depth",k=k,phase_rad=p))
    for k,p in ((26,1.21),(30,-.33)): dev.append(base(f"rl_dev_ehigh_{k}","dev","single",axis="east",k=k,phase_rad=p))
    for suffix,c,p in (("a",(20,41),.91),("b",(25,44),-1.73)): dev.append(base(f"rl_dev_mixed_{suffix}","dev","mixed",axis="east_depth",switch_times_s=list(c),phase_rad=p))
    lock=[]
    for k,p in ((5,-2.04),(7,1.36),(28,.37),(32,-1.94)): lock.append(base(f"rl_lock_{'low' if k<10 else 'dhigh'}_{k}","rl_test_locked_v1","single",axis="depth",k=k,phase_rad=p))
    for k,p in ((28,2.11),(32,-.68)): lock.append(base(f"rl_lock_ehigh_{k}","rl_test_locked_v1","single",axis="east",k=k,phase_rad=p))
    for suffix,c,p in (("a",(18,43),1.89),("b",(24,47),-2.62)): lock.append(base(f"rl_lock_mixed_{suffix}","rl_test_locked_v1","mixed",axis="east_depth",switch_times_s=list(c),phase_rad=p))
    lists={"tasks_train.jsonl":train,"tasks_dev.jsonl":dev,"tasks_rl_test_locked_v1.jsonl":lock}
    hashes=[]
    for name,rows in lists.items():
      payload=b"".join(canonical(x)+b"\n" for x in rows); path=out/name
      if path.exists(): raise FileExistsError(f"refusing to overwrite frozen task list: {path}")
      path.write_bytes(payload); hashes.extend(digest(canonical(x)) for x in rows)
    if len(set(hashes)) != len(hashes): raise ValueError("duplicate task definitions across splits")
    protocol={"version":"gs12_rl_beta_pilot_v1","seed":20261111,"train_count":64,"dev_count":8,"locked_count":8,"action_betas":[0,.25,.5,.75],"obs_dim":19,"reward":"segment_reward_v1","task_sha256":{n:digest((out/n).read_bytes()) for n in lists}}
    p=out/"protocol.json"
    if p.exists(): raise FileExistsError(f"refusing to overwrite: {p}")
    p.write_bytes(canonical(protocol)+b"\n")
    return protocol

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("prepare",choices=["prepare"]); ap.add_argument("--out",required=True); a=ap.parse_args(); print(json.dumps(freeze(a.out),indent=2))
