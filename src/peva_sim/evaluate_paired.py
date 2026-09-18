"""Baseline evaluator; refuses heuristic-guidance checkpoints and verifies provenance."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .env import Config
from .real_env import RealMaritimeSAR
from .mappo_v2 import Model,features,critic_state,ROOT
from .ppo_math import disk_action
from .protocol import file_sha
from .scenario_manifest import validate_windows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True); ap.add_argument("--scenarios",required=True)
    ap.add_argument("--out",required=True); a=ap.parse_args()
    torch.set_num_threads(1)
    cp=(ROOT/a.checkpoint).resolve(); scenario_path=(ROOT/a.scenarios).resolve()
    out=(ROOT/a.out).resolve()
    if not out.is_relative_to(ROOT): ap.error("Output outside project")
    if out.exists(): ap.error("Refusing overwrite")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(cp,map_location=device,weights_only=False); meta=ck["metadata"]
    if meta["args"].get("guidance",0)!=0:
        ap.error("Heuristic-guidance checkpoint is not an audited on-policy baseline")
    scenarios=json.loads(scenario_path.read_text()); validate_windows(scenarios["episodes"])
    data=ROOT/meta["args"]["data"]; split=ROOT/meta["args"]["manifest"]
    if file_sha(data)!=scenarios["data_sha256"] or file_sha(split)!=scenarios["split_sha256"]:
        ap.error("Scenario provenance mismatch")
    if meta["manifest_sha256"]!=scenarios["split_sha256"]:
        ap.error("Checkpoint used a different data partition")
    c=Config(**meta["config"])
    feature_fn=(__import__("peva_sim.peva_features",fromlist=["PEVAFeatureExtractor"]).PEVAFeatureExtractor(meta["args"]["peva_aux"],c,meta["args"]["pvf"]) if meta["args"].get("peva_aux") else lambda z: features(z,c,meta["args"]["pvf"]))
    env=RealMaritimeSAR(data,c,split=scenarios["split"],manifest_path=split)
    adapter=None
    if meta["args"].get("transport_aux"):
        from .transport_features import TransportFeatureAdapter
        auxpath=ROOT/meta["args"]["transport_aux"]
        if file_sha(auxpath)!=meta["auxiliary_sha256"]:
            raise ValueError("auxiliary checkpoint changed since policy training")
        adapter=TransportFeatureAdapter(auxpath,c,seed=0,message_weighting=meta.get("message_weighting","uniform"))
        adapter.model.load_state_dict(ck["auxiliary_model"])
        feature_fn=lambda z: adapter.step(env,z)
    model=Model(ck["model"]["actor.0.weight"].shape[1],
                ck["model"]["critic.0.weight"].shape[1]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    out.mkdir(parents=True,exist_ok=False)
    rows=[]; started=time.perf_counter()
    for row in scenarios["episodes"]:
        index=row["replay_index"]
        ds=env.replay.dataset; gi=env.replay.indices[index]
        if int(gi)!=row["global_index"] or str(ds.track_id[gi])!=row["track_id"]:
            raise ValueError("Scenario membership mismatch")
        obs,_=env.reset(seed=row["environment_seed"],episode_index=index)
        if adapter: adapter.rng.manual_seed(row["environment_seed"]+123000)
        if env.start_utc!=row["start_utc"]: raise ValueError("Scenario time mismatch")
        total=0.; steps=0
        while not env.done:
            with torch.no_grad():
                x=torch.as_tensor(feature_fn(obs),device=device)
                mu,_=model.policy(x)
                action=disk_action(mu,c.speed_mps).cpu().numpy()
            obs,r,term,trunc,info=env.step(action); total+=r; steps+=1
        record=dict(row,**info,steps=steps,return_value=total)
        rows.append(record)
        with (out/"episodes.jsonl").open("a") as f: f.write(json.dumps(record)+"\n")
        print(json.dumps({"completed":len(rows),"total":len(scenarios["episodes"])}),flush=True)
    summary={"checkpoint":str(cp),"checkpoint_sha256":file_sha(cp),
             "algorithm_version":meta["version"],
             "auxiliary_sha256":meta.get("auxiliary_sha256"),
             "auxiliary_randomness":"dedicated generator, seed=environment_seed+123000" if adapter else None,
             "scenarios_sha256":file_sha(scenario_path),"split_sha256":file_sha(split),
             "data_sha256":file_sha(data),"split":scenarios["split"],"episodes":rows,
             "policy_seed":meta["args"]["seed"],"pvf":meta["args"]["pvf"],
             "particles":c.particles,"wall_s":time.perf_counter()-started,
             "status":"pilot evaluation, not converged publication performance"}
    for key in ["success_rate","ttd_all","survival_score"]:
        summary["mean_"+key]=float(np.mean([r[key] for r in rows]))
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); env.close()
if __name__=="__main__": main()
