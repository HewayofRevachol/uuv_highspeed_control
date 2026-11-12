#!/usr/bin/env python3
"""Add 0.003-amplitude triplets for every constrained low-speed full-amplitude band."""
import importlib.util
import json
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
RUNNER_PATH = REPO / "experiments/gs12_analaysis/frequency_extension_src/run_frequency_extension.py"
sys.path.insert(0, str(RUNNER_PATH.parent))
spec=importlib.util.spec_from_file_location("gs12_runner",RUNNER_PATH)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def main():
    runs=[json.loads(p.read_text()) for p in (ROOT/"runs").glob("p*_v*.json")]
    needed=set()
    for r in runs:
        if r.get("amplitude_per_line") != .006 or r.get("phase_id") != "primary" or r.get("speed_mps")==15:
            continue
        if r.get("status") != "complete_constrained": continue
        if r.get("axis") == -1:
            for band in ("B1","B2","B3","B4"):
                needed.add((r["speed_mps"],r["condition"],band))
        else:
            needed.add((r["speed_mps"],r["condition"],r["band"]))
    jobs=sorted((s,c,b,a) for s,c,b in needed for a in range(3))
    p=argparse.ArgumentParser();p.add_argument("--start",type=int,default=0);p.add_argument("--stop",type=int)
    ns=p.parse_args();stop=len(jobs) if ns.stop is None else min(ns.stop,len(jobs))
    args=SimpleNamespace(repo=REPO,out=ROOT)
    for i,(s,c,b,a) in enumerate(jobs[ns.start:stop],ns.start+1):
        rec=runner.execute(args,s,c,b,a,.003,"primary")
        runner.update_manifest(ROOT,rec,"other-speeds-half-constrained",i)
        print(json.dumps({"stage":"other-speeds-half-constrained","i":i,"n":len(jobs),
                          "speed_mps":s,"condition":c,"band":b,"axis":a,
                          "status":rec["status"],"run_id":rec["run_id"]}),flush=True)
    print(json.dumps({"half_amplitude_groups":len(needed),"runs":len(jobs),"executed":[ns.start,stop],
                      "groups":[list(x) for x in sorted(needed)]}),flush=True)


if __name__=="__main__": main()
