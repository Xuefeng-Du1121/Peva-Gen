"""Fixed, finite validation-only calibration matrix; never an automatic tuner."""
import argparse,itertools,json,subprocess,time
from pathlib import Path
import torch
from datetime import datetime,timezone
from .protocol import file_sha
from .evaluation_invariants import verify_episode
ROOT=Path(__file__).resolve().parents[2]
SCENARIOS=ROOT/"configs/real-validation-scenarios-v1.json"

def matrix(smoke=False):
    return [dict(name=f"{method}-lambda{lam}-entropy{entropy}",
                 method=method,gae_lambda=lam,entropy_coef=entropy,
                 seed=0,updates=2 if smoke else 200,rollout=4 if smoke else 256)
            for lam,entropy,method in itertools.product((.95,1.),(0.,.01),("mappo","pvf"))]

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args()
    out=(ROOT/a.out).resolve()
    if not out.is_relative_to(ROOT): ap.error("Output must stay under project")
    scenarios=json.loads(SCENARIOS.read_text())
    if scenarios["split"]!="validation": raise ValueError("Validation only")
    out.mkdir(parents=True,exist_ok=False)
    jobs=matrix(a.smoke)
    plan=dict(created_utc=now(),scope="single-seed diagnostic; not final baseline selection",
              smoke=a.smoke,max_concurrent_trainers=2,scenarios_sha256=file_sha(SCENARIOS),
              selection_rule="Report all eight cells. Do not select on test results. Any chosen setting needs paired multi-seed validation.",
              stopping_rule="Fixed budget for every cell; failures are retained; no adaptive extensions.",
              source_sha256=file_sha(__file__),trainer_sha256=file_sha(ROOT/"src/peva_sim/mappo_calibration.py"),
              jobs=jobs)
    (out/"plan.json").write_text(json.dumps(plan,indent=2))
    status=dict(status="running",jobs={},started_utc=now())
    def save():
        temp=out/"status.tmp"
        temp.write_text(json.dumps(status,indent=2));temp.replace(out/"status.json")
    save()
    for begin in range(0,len(jobs),2):
        children=[]
        for job in jobs[begin:begin+2]:
            command=["bash","run.sh","-m","peva_sim.mappo_calibration","--real-data",
                     "--coverage-mode","intensity","--updates",str(job["updates"]),
                     "--rollout",str(job["rollout"]),"--seed","0",
                     "--gae-lambda",str(job["gae_lambda"]),"--entropy-coef",str(job["entropy_coef"]),
                     "--snapshot-every",str(1 if a.smoke else 50),"--out",str(out/job["name"])]
            if job["method"]=="pvf": command.append("--pvf")
            log=(out/(job["name"]+".log")).open("x")
            p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            children.append((job,p,log))
            status["jobs"][job["name"]]=dict(status="training",pid=p.pid,command=command)
            save()
        for job,p,log in children:
            rc=p.wait();log.close()
            status["jobs"][job["name"]].update(status="trained" if rc==0 else "failed",exit_code=rc)
            save()
    if any(r["status"]=="failed" for r in status["jobs"].values()):
        status.update(status="failed",finished_utc=now());save();raise SystemExit(1)
    rows=[]
    for job in jobs:
        ck=torch.load(out/job["name"]/"checkpoint.pt",map_location="cpu",weights_only=False)
        if ck["environment_steps"]!=job["updates"]*job["rollout"]: raise ValueError("Budget mismatch")
        args=ck["metadata"]["args"]
        for key in ("seed","updates","rollout","gae_lambda","entropy_coef"):
            if args[key]!=job[key]: raise ValueError("Training argument mismatch: "+key)
        if args["pvf"]!=(job["method"]=="pvf"): raise ValueError("Method mismatch")
        if not all(torch.isfinite(v).all() for v in ck["model"].values()): raise ValueError("Nonfinite model")
        if ck["metadata"]["source_sha256"]!=plan["trainer_sha256"]: raise ValueError("Trainer changed")
    if file_sha(ROOT/"src/peva_sim/mappo_calibration.py")!=plan["trainer_sha256"]: raise ValueError("Trainer changed")
    if not a.smoke:
        for job in jobs:
            name=job["name"];checkpoint=out/name/"checkpoint.pt";evaluation=out/(name+"-validation")
            if file_sha(SCENARIOS)!=plan["scenarios_sha256"]: raise ValueError("Scenarios changed")
            command=["bash","run.sh","-m","peva_sim.evaluate_paired","--checkpoint",str(checkpoint),
                     "--scenarios",str(SCENARIOS),"--out",str(evaluation)]
            with (out/(name+"-validation.log")).open("x") as log:
                p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                status["jobs"][name].update(status="evaluating",evaluation_pid=p.pid);save()
                rc=p.wait()
            if rc:
                status["jobs"][name].update(status="evaluation_failed",evaluation_exit_code=rc)
                status.update(status="failed",finished_utc=now())
                save()
                raise SystemExit(rc)
            summary=json.loads((evaluation/"summary.json").read_text())
            config=json.loads((out/name/"metadata.json").read_text())["config"]
            episodes=summary["episodes"]
            if len(episodes)!=len(scenarios["episodes"]): raise ValueError("Incomplete evaluation")
            for record,scenario in zip(episodes,scenarios["episodes"]): verify_episode(record,scenario,config)
            row=dict(job,checkpoint_sha256=file_sha(checkpoint),episodes=len(episodes))
            for key in ("success_rate","ttd_all","survival_score"):
                row[key]=sum(r[key] for r in episodes)/len(episodes)
                if abs(row[key]-summary["mean_"+key])>1e-10: raise ValueError("Summary mismatch")
            rows.append(row)
            status["jobs"][name].update(status="verified");save()
    (out/"results.json").write_text(json.dumps(dict(scope=plan["scope"],smoke=a.smoke,rows=rows),indent=2))
    status.update(status="completed",finished_utc=now());save()
    print(json.dumps(status),flush=True)

if __name__=="__main__": main()
