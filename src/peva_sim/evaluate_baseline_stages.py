"""Evaluate every predeclared stage after paired training passes integrity audit."""
import argparse,json,subprocess
from pathlib import Path
from .audit_baseline_pair import audit,ROOT
from .protocol import file_sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pair",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--scenarios",default="configs/real-validation-scenarios-v1.json")
    ap.add_argument("--plan-only",action="store_true")
    a=ap.parse_args()
    output=(ROOT/a.out).resolve();scenarios=(ROOT/a.scenarios).resolve()
    if not output.is_relative_to(ROOT) or not scenarios.is_relative_to(ROOT):
        ap.error("paths outside project")
    report=audit(ROOT/a.pair)
    protocol=json.loads(scenarios.read_text())
    if protocol["split"]!="validation": ap.error("learning curves must not select on test")
    jobs=[]
    for method,result in report["methods"].items():
        for checkpoint in result["checkpoints"]:
            destination=output/method/f"step-{checkpoint['steps']:09d}"
            jobs.append({"method":method,**checkpoint,"out":str(destination),
                         "command":["bash","run.sh","-m","peva_sim.evaluate_paired",
                                    "--checkpoint",checkpoint["path"],"--scenarios",str(scenarios),"--out",str(destination)]})
    output.mkdir(parents=True,exist_ok=False)
    plan={"purpose":"predeclared matched-budget validation curves, no test-set selection",
          "scenario_sha256":file_sha(scenarios),"audit":report,"jobs":jobs,
          "status":"plan_only" if a.plan_only else "running"}
    def save():
        temporary=output/"plan.tmp"
        temporary.write_text(json.dumps(plan,indent=2));temporary.replace(output/"plan.json")
    save()
    if a.plan_only:
        print(json.dumps({"jobs":len(jobs),"status":"plan_only"}));return
    try:
        for job in jobs:
            if file_sha(job["path"])!=job["sha256"]: raise ValueError("checkpoint changed after audit")
            with (output/f"{job['method']}-{job['steps']}.log").open("x") as log:
                result=subprocess.run(job["command"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError("stage evaluation failed: "+str(job["out"]))
            summary=json.loads((Path(job["out"])/"summary.json").read_text())
            if summary["checkpoint_sha256"]!=job["sha256"] or summary["scenarios_sha256"]!=plan["scenario_sha256"]:
                raise ValueError("evaluated source mismatch")
            if len(summary["episodes"])!=len(protocol["episodes"]): raise ValueError("incomplete evaluation")
            job["result"]={key:summary[key] for key in ("mean_success_rate","mean_ttd_all","mean_survival_score")}
            job["status"]="completed";save()
    except Exception as error:
        plan["status"]="failed";plan["error"]=str(error);save();raise
    plan["status"]="completed";save()
    print(json.dumps({"status":"completed","jobs":len(jobs)}),flush=True)
if __name__=="__main__": main()
