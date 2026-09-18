"""Run a fixed matched-budget baseline pair; status is derived from child exits."""
import argparse,json,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True);ap.add_argument("--updates",type=int,default=200)
    ap.add_argument("--rollout",type=int,default=256);ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--snapshot-every",type=int,default=50)
    a=ap.parse_args()
    if min(a.updates,a.rollout,a.snapshot_every)<1: ap.error("positive counts required")
    out=(ROOT/a.out).resolve()
    if not out.is_relative_to(ROOT): ap.error("output outside project")
    out.mkdir(parents=True,exist_ok=False)
    def now(): return datetime.now(timezone.utc).isoformat()
    state={"status":"starting","purpose":"matched learning-curve pilot; not formal convergence evidence",
           "created_utc":now(),"budget_steps_per_run":a.updates*a.rollout,
           "seed":a.seed,"selection":"fixed seed/budget; no test-set selection","runs":{}}
    def save():
        tmp=out/"status.tmp"
        tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/"status.json")
    processes={};handles=[]
    try:
        for method in ("mappo","pvf"):
            command=["bash","run.sh","-m","peva_sim.mappo_v2","--real-data",
                     "--coverage-mode","intensity","--updates",str(a.updates),"--rollout",str(a.rollout),
                     "--seed",str(a.seed),"--snapshot-every",str(a.snapshot_every),
                     "--out",str(out/method)]
            if method=="pvf": command.append("--pvf")
            log=(out/(method+".log")).open("x")
            handles.append(log)
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            processes[method]=process
            state["runs"][method]={"pid":process.pid,"command":command,"status":"running","started_utc":now()}
            save()
        state["status"]="running";save()
        while any(p.poll() is None for p in processes.values()):
            changed=False
            for method,process in processes.items():
                code=process.poll()
                if code is not None and state["runs"][method]["status"]=="running":
                    state["runs"][method].update(status="completed" if code==0 else "failed",
                                                 exit_code=code,finished_utc=now())
                    changed=True
            if changed: save()
            time.sleep(2)
        for method,process in processes.items():
            code=process.returncode
            state["runs"][method].update(status="completed" if code==0 else "failed",exit_code=code,finished_utc=now())
        state["status"]="completed" if all(p.returncode==0 for p in processes.values()) else "failed"
        state["finished_utc"]=now();save()
        print(json.dumps(state),flush=True)
        if state["status"]!="completed": raise SystemExit(1)
    finally:
        for handle in handles: handle.close()
if __name__=="__main__": main()
