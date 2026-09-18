"""Fixed development replication; never selects checkpoints or touches test cases."""
import json, os, subprocess, time
from pathlib import Path
from .protocol import file_sha
ROOT=Path(__file__).resolve().parents[2]

def main():
    os.chdir(ROOT)
    out=ROOT/"runs/separate-replication-seeds12-20260913"
    out.mkdir(exist_ok=False)
    inventory="runs/ocean-forcing-expanded-development-inventory-v2.json"
    sources={str(p.relative_to(ROOT)):file_sha(p) for p in sorted((ROOT/"src/peva_sim").glob("*.py"))}
    sources[inventory]=file_sha(ROOT/inventory)
    jobs=[]
    for seed in (1,2):
        for lam in (.95,1.):
            for pvf in (False,True):
                name=f"seed{seed}-lambda{lam}-{'pvf' if pvf else 'mappo'}"
                train=str((out/name).relative_to(ROOT))
                evaluate=train+"-validation"
                command=["bash","run.sh","-m","peva_sim.train_separate_mappo",
                         "--inventory",inventory,"--out",train,"--seed",str(seed),
                         "--updates","200","--rollout","256","--epochs","4",
                         "--minibatch","64","--gae-lambda",str(lam),"--entropy-coef","0.01"]
                if pvf: command.append("--pvf")
                jobs.append(dict(name=name,train=command,evaluate=["bash","run.sh","-m",
                    "peva_sim.evaluate_simulated","--checkpoint",train+"/checkpoint.pt",
                    "--inventory",inventory,"--out",evaluate,"--episodes-per-scenario","10"]))
    manifest=dict(scope="development replication, NOT formal results",training_seeds=[1,2],
        steps_per_cell=51200,selection="all final checkpoints, no selection",
        validation="same 20 scenarios/seeds as seed0; test untouched by this batch",
        source_sha256=sources,jobs=jobs)
    (out/"plan.json").write_text(json.dumps(manifest,indent=2))
    def event(**row):
        with (out/"events.jsonl").open("a") as f:
            f.write(json.dumps(dict(time=time.time(),driver_pid=os.getpid(),**row))+"\n")
    event(status="started")
    for job in jobs:
        for stage in ("train","evaluate"):
            if any(file_sha(ROOT/p)!=sha for p,sha in sources.items()):
                event(status="failed",reason="source or inventory changed")
                raise RuntimeError("Source changed: refuse mixed-version replication")
            with (out/(job["name"]+"-"+stage+".log")).open("x") as log:
                child=subprocess.Popen(job[stage],stdout=log,stderr=subprocess.STDOUT)
                event(status="running",job=job["name"],stage=stage,child_pid=child.pid)
                code=child.wait()
            event(status="completed" if code==0 else "failed",job=job["name"],stage=stage,returncode=code)
            if code: raise RuntimeError(f"{job['name']} {stage} failed: {code}")
    event(status="completed",scope="all eight training and validation cells")
if __name__=="__main__": main()
