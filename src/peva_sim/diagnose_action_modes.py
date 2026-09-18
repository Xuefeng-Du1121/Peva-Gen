"""Paired action-mode/positive-control diagnostic; never updates a policy."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .env import Config,segment_distance
from .real_env import RealMaritimeSAR
from .mappo_v2 import Model,features
from .ppo_math import disk_action
from .protocol import file_sha
from .audit_baseline_pair import audit,ROOT
from .evaluation_invariants import verify_episode


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--pair",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--scenarios",default="configs/real-validation-scenarios-v1.json")
    a=ap.parse_args();out=(ROOT/a.out).resolve();scenario_path=(ROOT/a.scenarios).resolve()
    if not out.is_relative_to(ROOT) or out.exists(): ap.error("invalid output or refusing overwrite")
    provenance=audit(ROOT/a.pair)
    scenarios=json.loads(scenario_path.read_text())
    if scenarios["split"]!="validation": ap.error("diagnostics use validation, not test")
    models={};configs={};metadata={}
    for name in ("mappo","pvf"):
        path=Path(provenance["methods"][name]["checkpoints"][-1]["path"])
        ck=torch.load(path,weights_only=False,map_location="cpu");meta=ck["metadata"]
        model=Model(ck["model"]["actor.0.weight"].shape[1],ck["model"]["critic.0.weight"].shape[1])
        model.load_state_dict(ck["model"]);model.eval();models[name]=model;metadata[name]=meta
        configs[name]=Config(**meta["config"])
    torch.set_num_threads(1)
    c=configs["mappo"];meta=metadata["mappo"]
    if file_sha(ROOT/meta["args"]["data"])!=scenarios["data_sha256"] or file_sha(ROOT/meta["args"]["manifest"])!=scenarios["split_sha256"]:
        raise ValueError("scenario/data source mismatch")
    env=RealMaritimeSAR(ROOT/meta["args"]["data"],c,split="validation",manifest_path=ROOT/meta["args"]["manifest"])
    out.mkdir(parents=True,exist_ok=False)
    report={"scope":"one-seed paired diagnostic; not a selected evaluation protocol or formal superiority result",
            "source_sha256":file_sha(__file__),"scenario_sha256":file_sha(scenario_path),"audit":provenance,
            "policy_sampling_seed":"environment_seed+1729","methods":{}}
    started=time.perf_counter()
    try:
        for name in ("mappo","pvf","public_datum"):
            records=[]
            for scenario in scenarios["episodes"]:
                obs,_=env.reset(scenario["environment_seed"],episode_index=scenario["replay_index"])
                gi=int(env.replay.indices[scenario["replay_index"]])
                if gi!=scenario["global_index"] or env.start_utc!=scenario["start_utc"]:
                    raise ValueError("scenario identity mismatch")
                generator=torch.Generator().manual_seed(scenario["environment_seed"]+1729)
                total=0.;steps=0;minimum=float("inf");within=0;heading=[]
                while not env.done:
                    positions=obs["positions_m"].copy()
                    datum_delta=c.region_m/2-positions
                    if name=="public_datum":
                        distance=np.linalg.norm(datum_delta,axis=-1,keepdims=True)
                        action=datum_delta/np.maximum(distance,1e-12)*np.minimum(c.speed_mps,distance/c.dt_s)
                    else:
                        with torch.no_grad():
                            mu,ls=models[name].policy(torch.tensor(features(obs,c,name=="pvf")))
                            raw=mu+ls.exp()*torch.randn(mu.shape,generator=generator)
                            action=disk_action(raw,c.speed_mps).numpy()
                    denominator=np.linalg.norm(action,axis=-1)*np.linalg.norm(datum_delta,axis=-1)
                    heading.append(float(np.mean((action*datum_delta).sum(-1)/np.maximum(denominator,1e-12))))
                    obs,reward,_,_,info=env.step(action);total+=reward;steps+=1
                    # Privileged coordinates are read ONLY after action, for diagnostics.
                    distance=float(segment_distance(env.targets,positions,env.uavs).min())
                    minimum=min(minimum,distance);within+=int(distance<=c.sensing_m)
                record=dict(scenario,**info,steps=steps,return_value=total,
                            minimum_swept_target_distance_m=minimum,steps_with_target_in_footprint=within,
                            mean_heading_cosine_to_public_datum=float(np.mean(heading)))
                verify_episode(record,scenario,env.config_dict())
                records.append(record)
                with (out/(name+".jsonl")).open("a") as f: f.write(json.dumps(record)+"\n")
                print(json.dumps({"method":name,"completed":len(records),"total":len(scenarios["episodes"])}),flush=True)
            report["methods"][name]={"action_mode":"public-datum control" if name=="public_datum" else "sampled Gaussian policy",
                "episodes":records,"mean_success_rate":float(np.mean([r["success_rate"] for r in records])),
                "mean_ttd_all":float(np.mean([r["ttd_all"] for r in records])),
                "mean_survival_score":float(np.mean([r["survival_score"] for r in records]))}
        report["wall_s"]=time.perf_counter()-started
        (out/"summary.json").write_text(json.dumps(report,indent=2))
    finally: env.close()
if __name__=="__main__": main()
