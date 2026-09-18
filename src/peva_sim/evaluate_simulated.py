"""Unified simulated validation for ocean baselines and multi-target coupled policies."""
import argparse,json,time,platform
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from .ocean_training_pool import OceanTrainingPool
from .ocean_forced_env import OceanForcedSAR
from .forcing_scenarios import support,validate_split_support
from .mappo_v2 import Model,features
from .multitarget_features import MultiTargetFeatureAdapter
from .ppo_math import disk_action
from .protocol import file_sha
from .audit_simulated_collection import verify_episode_arrays
from .evaluation_trace import audit_trace
from .confirmed_environment import ConfirmedEnvironment
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True);ap.add_argument("--inventory",required=True)
    ap.add_argument("--out",required=True);ap.add_argument("--episodes-per-scenario",type=int,default=10)
    ap.add_argument("--packet-drop",type=float,default=None)
    ap.add_argument("--radio-range-m",type=float,default=None)
    a=ap.parse_args()
    if a.packet_drop is not None and (not np.isfinite(a.packet_drop) or not 0<=a.packet_drop<=1):
        ap.error("packet-drop must be within [0,1]")
    if a.radio_range_m is not None and (not np.isfinite(a.radio_range_m) or a.radio_range_m<=0):
        ap.error("radio-range-m must be positive")
    cp,inv,out=[(ROOT/v).resolve() for v in (a.checkpoint,a.inventory,a.out)]
    if not all(p.is_relative_to(ROOT) for p in (cp,inv,out)) or a.episodes_per_scenario<1:
        ap.error("Invalid paths/counts")
    if out.exists(): ap.error("Refusing overwrite")
    torch.set_num_threads(1)
    ck=torch.load(cp,map_location="cpu",weights_only=False);meta=ck["metadata"]
    allowed=("mappo_ocean_forced_v1","mappo_ocean_pool_v1","simulated_multitarget_coupled_v1")
    allowed += ("mappo_separate_optim_v1","simulated_multitarget_confirmed_v2")
    if meta["version"] not in allowed: ap.error("Unsupported checkpoint architecture")
    pool=OceanTrainingPool(inv);c=pool.cfg
    if meta["config"]!=pool.config_dict(): raise ValueError("Evaluation environment differs from training")
    overrides={k:v for k,v in (("packet_drop",a.packet_drop),("radio_range_m",a.radio_range_m)) if v is not None}
    c=replace(c,**overrides)
    inventory=json.loads(inv.read_text())
    cases=[r for r in inventory["scenarios"] if r["split"]=="validation"]
    if not cases: raise ValueError("No validation cases")
    forcing=meta["forcing"]
    if "inventory_path" in forcing:
        source_path=Path(forcing["inventory_path"]).resolve()
        if not source_path.is_relative_to(ROOT) or file_sha(source_path)!=forcing["inventory_sha256"]:
            raise ValueError("Training inventory changed")
        source=json.loads(source_path.read_text())
        if source["field_sha256"]!=inventory["field_sha256"]: raise ValueError("Forcing source mismatch")
        train=[r for r in source["scenarios"] if r["split"]=="train"]
    else:
        if meta["field_sha256"]!=inventory["field_sha256"]: raise ValueError("Forcing source mismatch")
        train=[dict(id="checkpoint-training-patch",split="train",
                    support=support(pool.field,forcing["origin_lonlat"],forcing["start_utc"],c.region_m,c.horizon_s))]
    validate_split_support(train+cases)
    model=Model(ck["model"]["actor.0.weight"].shape[1],ck["model"]["critic.0.weight"].shape[1])
    model.load_state_dict(ck["model"]);model.eval()
    adapter=None
    confirmed=meta["version"]=="simulated_multitarget_confirmed_v2"
    if confirmed and meta.get("confirmation_protocol")!=ConfirmedEnvironment.protocol_version:
        raise ValueError("Unknown confirmation protocol")
    if meta["version"] in ("simulated_multitarget_coupled_v1","simulated_multitarget_confirmed_v2"):
        aux=(ROOT/meta["args"]["transport_aux"]).resolve()
        if not aux.is_relative_to(ROOT) or file_sha(aux)!=meta["auxiliary_sha256"]:
            raise ValueError("Auxiliary checkpoint changed")
        adapter=MultiTargetFeatureAdapter(aux,c,message_weighting=meta["message_weighting"])
        adapter.model.load_state_dict(ck["auxiliary_model"])
        adapter.model.eval()
    out.mkdir(parents=True,exist_ok=False)
    source_manifest={p.name:file_sha(p) for p in sorted((ROOT/"src/peva_sim").glob("*.py"))}
    (out/"source-manifest.json").write_text(json.dumps(source_manifest,indent=2))
    evaluation_started=time.perf_counter()
    rows=[]
    for case_index,case in enumerate(cases):
        env=OceanForcedSAR(pool.field,case["origin_lonlat"],case["start_utc"],c)
        if confirmed: env=ConfirmedEnvironment(env)
        for repeat in range(a.episodes_per_scenario):
            seed=820000+1000*case_index+repeat
            obs,_=env.reset(seed)
            if adapter: adapter.rng.manual_seed(seed+123000)
            times=[];found=[];total=0.
            actions=[];rewards=[];positions=[];targets=[];inference_ms=[]
            episode_started=time.perf_counter()
            while not env.done:
                times.append(env.t);found.append(env.found.copy())
                state=env.state()
                positions.append(state["uavs_m"].copy())
                targets.append(state["targets_m"].copy())
                inference_started=time.perf_counter()
                with torch.no_grad():
                    x=adapter.step(env,obs) if adapter else features(obs,c,meta["args"]["pvf"])
                    mu,_=model.policy(torch.tensor(x,dtype=torch.float32))
                    action=disk_action(mu,c.speed_mps).numpy()
                inference_ms.append((time.perf_counter()-inference_started)*1000)
                obs,reward,_,_,info=env.step(action);total+=reward
                actions.append(action.copy());rewards.append(reward)
            record=dict(scenario_id=case["id"],seed=seed,steps=len(times),return_value=total,info=info)
            record["runtime"]=dict(episode_wall_s=time.perf_counter()-episode_started,
                inference_mean_ms=float(np.mean(inference_ms)),inference_p95_ms=float(np.percentile(inference_ms,95)),
                inference_total_ms=float(np.sum(inference_ms)))
            audit_record=dict(info={**info,"bytes":dict(info["bytes"])})
            if not adapter:
                audit_record["info"]["bytes"].update(peer_payload=len(times)*c.n_uavs*42,
                                                     peer_header=len(times)*c.n_uavs*c.header_bytes)
                if info["bytes"]["peer_payload"] or info["bytes"]["peer_header"]:
                    raise ValueError("Unexpected baseline peer traffic")
            if confirmed:
                post=np.concatenate((np.asarray(found)[1:],env.found[None]),axis=0)
                extra=int(len(times)*4+post.sum()*4)
                if env.ledger.received_payload_bytes!=extra:
                    raise ValueError("Confirmation payload accounting mismatch")
                audit_record["info"]["bytes"]["shared_downlink"]-=extra
            verify_episode_arrays(np.asarray(times),np.asarray(found),audit_record,vars(c))
            if not np.isclose(total,info["survival_score"]*c.n_targets,atol=1e-10):
                raise ValueError("Reward/survival mismatch")
            trace=out/("episode-%04d.npz"%len(rows))
            np.savez_compressed(trace,time_s=np.asarray(times),found=np.asarray(found),
                                terminal_found=env.found.copy(),actions_mps=np.asarray(actions),
                                rewards=np.asarray(rewards),uavs_m=np.asarray(positions),
                                targets_m=np.asarray(targets),inference_ms=np.asarray(inference_ms))
            record["trace_file"]=trace.name
            record["trace_sha256"]=file_sha(trace)
            with np.load(trace) as saved:
                audit_trace(saved,record,vars(c))
            rows.append(record)
            with (out/"episodes.jsonl").open("a") as f: f.write(json.dumps(record)+"\n")
            print(json.dumps(dict(completed=len(rows),total=len(cases)*a.episodes_per_scenario)),flush=True)
    result=dict(status="development validation, not formal performance",checkpoint_sha256=file_sha(cp),
                inventory_sha256=file_sha(inv),source_sha256=file_sha(__file__),algorithm=meta["version"],
                training_steps=ck["environment_steps"],episodes=len(rows),
                scenarios=[r["id"] for r in cases],records=rows)
    result["runtime"]=dict(wall_s=time.perf_counter()-evaluation_started,
        device="cpu",torch_threads=torch.get_num_threads(),python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
        timing_scope="CPU policy features, peer exchange and inference; excludes env.step and audit IO",
        source_manifest_sha256=file_sha(out/"source-manifest.json"))
    result["evaluation_config"]=vars(c)
    result["robustness_overrides"]=overrides
    result["shared_link_assumption"]="unchanged reliable zero-delay; overrides affect peer radio only"
    for key in ("success_rate","ttd_all","survival_score"):
        result[key]=float(np.mean([r["info"][key] for r in rows]))
    (out/"summary.json").write_text(json.dumps(result,indent=2));pool.close()

if __name__=="__main__": main()
