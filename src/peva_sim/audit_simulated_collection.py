"""Completion audit for simulated collection; never treats a live job as complete."""
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from .protocol import file_sha
from .ocean_training_pool import OceanTrainingPool
from .collect_simulated_belief import TARGET_SCHEMA
from .collection_policy import CollectionPolicy
ROOT=Path(__file__).resolve().parents[2]

def verify_episode_arrays(times,found,record,config,peer_payload_bytes=42,
                          confirmation_snapshots=False):
    dt=config["dt_s"];horizon=config["horizon_s"];n=config["n_targets"]
    steps=len(times);info=record["info"]
    if not 1<=steps<=round(horizon/dt): raise ValueError("Episode length outside horizon")
    if not np.array_equal(times,np.arange(steps)*dt): raise ValueError("Episode time sequence mismatch")
    if found.shape!=(steps,n) or found.dtype!=np.bool_ or found[0].any():
        raise ValueError("Invalid initial found mask")
    if (np.diff(found.astype(int),axis=0)<0).any(): raise ValueError("Found target resurrected")
    if info["time_s"]!=steps*dt: raise ValueError("Terminal time mismatch")
    rescued=round(info["success_rate"]*n)
    if not math.isclose(info["success_rate"],rescued/n) or not 0<=rescued<=n:
        raise ValueError("Invalid target success fraction")
    known=found[-1].sum()
    if rescued<known or (rescued<n and steps*dt!=horizon): raise ValueError("Invalid termination")
    detection_times=[times[np.flatnonzero(found[:,j])[0]] for j in range(n) if found[:,j].any()]
    detection_times += [steps*dt]*(rescued-len(detection_times))
    expected_ttd=(sum(detection_times)+(n-rescued)*horizon)/(n*horizon)
    expected_survival=sum(math.exp(-t/config["survival_tau_s"]) for t in detection_times)/n
    for key,expected in (("ttd_all",expected_ttd),("survival_score",expected_survival)):
        if not math.isclose(info[key],expected,rel_tol=1e-9,abs_tol=1e-12):
            raise ValueError("Metric inconsistent with masks: "+key)
    if not isinstance(peer_payload_bytes,int) or not 0<=peer_payload_bytes<=config["payload_budget_bytes"]:
        raise ValueError("Invalid peer payload contract")
    confirmation_bytes=0
    if confirmation_snapshots:
        post_step_counts=list(found[1:].sum(axis=1))+[rescued]
        confirmation_bytes=sum(4+4*int(count) for count in post_step_counts)
    expected=dict(peer_payload=steps*config["n_uavs"]*peer_payload_bytes,
                  peer_header=(steps*config["n_uavs"]*config["header_bytes"]
                               if peer_payload_bytes else 0),
                  shared_uplink=steps*config["n_uavs"]*(16+config["header_bytes"])+rescued*4,
                  shared_downlink=(steps*(config["grid_side"]**2*4+8+config["header_bytes"])
                                   + confirmation_bytes))
    if info["bytes"]!=expected: raise ValueError("Communication mismatch")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args();source=(ROOT/a.samples).resolve();out=(ROOT/a.out).resolve()
    if not all(p.is_relative_to(ROOT) for p in (source,out)) or out.exists(): ap.error("Invalid paths/overwrite")
    meta=json.loads((source/"metadata.json").read_text())
    if meta["split"]!="train" or meta["target_schema"]!=TARGET_SCHEMA: raise ValueError("Wrong data contract")
    if file_sha(source/"samples.npz")!=meta["samples_sha256"]: raise ValueError("Data hash mismatch")
    pool=OceanTrainingPool(meta["forcing"]["inventory_path"])
    if pool.inventory_sha256!=meta["forcing"]["inventory_sha256"]: raise ValueError("Inventory changed")
    allowed={r["id"] for r in pool.training_scenarios};pool.close()
    records=meta["episodes"];counts=Counter();modes=defaultdict(set)
    c=meta["config"]
    with np.load(source/"samples.npz") as archive:
        for key in archive.files:
            if not np.isfinite(archive[key]).all(): raise ValueError("Nonfinite array: "+key)
        ids=archive["episode"];times=archive["time_s"];found=archive["found"];rows=len(ids)
        if len(records)!=meta["args"]["episodes"] or set(np.unique(ids))!=set(range(len(records))):
            raise ValueError("Incomplete episode count")
        expected_shapes=dict(context=(rows,c["n_uavs"],16+c["grid_side"]**2),target=(rows,2*c["n_targets"]),
                             particles=(rows,c["n_targets"],c["particles"],2),weights=(rows,c["n_targets"],c["particles"]),
                             received=(rows,c["n_uavs"],c["n_uavs"]),found=(rows,c["n_targets"]))
        for key,shape in expected_shapes.items():
            if archive[key].shape!=shape: raise ValueError("Shape mismatch: "+key)
        if archive["particles"].dtype!=np.float64: raise ValueError("Particle precision lost")
        weights=archive["weights"]
        if (weights<0).any(): raise ValueError("Negative particle mass")
        mass=weights.sum(-1)
        if (mass[found]!=0).any(): raise ValueError("Found target retains particle mass")
        if (np.abs(archive["target"])>1).any(): raise ValueError("Target outside region")
        for ep,record in enumerate(records):
            if record["episode"]!=ep or record["scenario_id"] not in allowed: raise ValueError("Episode/split mismatch")
            mask=ids==ep;positions=np.flatnonzero(mask)
            if not np.array_equal(positions,np.arange(positions[0],positions[-1]+1)): raise ValueError("Interleaved episode rows")
            verify_episode_arrays(times[mask],found[mask],record,c)
            if c["coverage_mode"]=="intensity":
                if not np.allclose(mass[mask][0],1.,rtol=0,atol=1e-10): raise ValueError("Initial mass mismatch")
                if (np.diff(mass[mask],axis=0)>1e-10).any(): raise ValueError("Undetected mass increased")
            counts[record["scenario_id"]]+=1;modes[record["scenario_id"]].add(record["behavior"])
    if meta["args"].get("balanced") and len(records)==3*len(allowed):
        if set(counts)!=allowed or set(counts.values())!={3}: raise ValueError("Unbalanced scene coverage")
        if any(value!=set(CollectionPolicy.MODES) for value in modes.values()): raise ValueError("Missing behavior coverage")
    report=dict(status="verified",samples_sha256=meta["samples_sha256"],rows=rows,episodes=len(records),
                scenarios=dict(counts),source_sha256=file_sha(__file__),scope="collection integrity, not model quality")
    with out.open("x") as f: json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)

if __name__=="__main__": main()
