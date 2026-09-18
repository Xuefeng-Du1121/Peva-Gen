"""Recompute completed learning-curve records independently of evaluator summaries."""
import argparse,json,math
from pathlib import Path
import torch
from .protocol import file_sha
from .evaluation_invariants import verify_episode
ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--evaluation",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args();directory=(ROOT/a.evaluation).resolve();out=(ROOT/a.out).resolve()
    if not directory.is_relative_to(ROOT) or not out.is_relative_to(ROOT) or out.exists():
        ap.error("invalid paths or refusing overwrite")
    plan=json.loads((directory/"plan.json").read_text())
    if plan["status"]!="completed": raise ValueError("stage evaluations incomplete")
    results=[];reference_scenarios=None
    for job in plan["jobs"]:
        if job.get("status")!="completed": raise ValueError("incomplete job")
        if file_sha(job["path"])!=job["sha256"]: raise ValueError("checkpoint changed")
        checkpoint=torch.load(job["path"],weights_only=False,map_location="cpu")
        config=checkpoint["metadata"]["config"]
        if config["n_targets"]!=1: raise ValueError("single-target audit only")
        args=job["command"];scenario_path=Path(args[args.index("--scenarios")+1])
        if file_sha(scenario_path)!=plan["scenario_sha256"]: raise ValueError("scenario source changed")
        scenarios=json.loads(scenario_path.read_text())["episodes"]
        if reference_scenarios is not None and scenarios!=reference_scenarios:
            raise ValueError("unpaired evaluation scenarios")
        reference_scenarios=scenarios
        folder=Path(job["out"])
        records=[json.loads(line) for line in (folder/"episodes.jsonl").read_text().splitlines()]
        summary=json.loads((folder/"summary.json").read_text())
        if len(records)!=len(scenarios) or summary["episodes"]!=records:
            raise ValueError("missing or inconsistent episode records")
        byte_totals=[verify_episode(r,s,config) for r,s in zip(records,scenarios)]
        row={"method":job["method"],"steps":job["steps"],"episodes":len(records),
             "mean_total_bytes":sum(byte_totals)/len(byte_totals),"checkpoint_sha256":job["sha256"]}
        for key in ("success_rate","ttd_all","survival_score"):
            value=sum(r[key] for r in records)/len(records)
            if not math.isclose(value,summary["mean_"+key],rel_tol=1e-9,abs_tol=1e-12):
                raise ValueError("incorrect summary: "+key)
            row[key]=value
        results.append(row)
    report={"status":"verified","source_sha256":file_sha(__file__),"scenario_sha256":plan["scenario_sha256"],
            "scope":"one-seed validation learning curves; not independent confirmatory test results",
            "rows":results}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=="__main__": main()
