"""Collect multi-target auxiliary labels from simulated TRAIN scenarios only.

Target slots retain simulator identities; found flags are recorded separately.
This is not a claim of permutation-invariant or control-sufficient encoding.
Float64 particle snapshots permit later decoder-location KDE gradients.
"""
import argparse,json
from pathlib import Path
import numpy as np
from .ocean_training_pool import OceanTrainingPool
from .belief_contract import swarm_context,CONTEXT_SCHEMA
from .collection_policy import CollectionPolicy
from .communication import encode_message
from .physics_priority import value_gradient
from .protocol import file_sha
ROOT=Path(__file__).resolve().parents[2]
TARGET_SCHEMA="simulated-multitarget-identity-slots-centered-region-v1"

def target_slots(env):
    target=np.asarray(env.targets)
    if target.shape!=(env.cfg.n_targets,2) or not np.isfinite(target).all():
        raise ValueError("Invalid multi-target state")
    if ((target<0)|(target>env.cfg.region_m)).any(): raise ValueError("Target outside domain")
    return (2*target/env.cfg.region_m-1).reshape(-1).astype(np.float32)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inventory",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--episodes",type=int,default=3);ap.add_argument("--seed",type=int,default=71)
    ap.add_argument("--balanced",action="store_true")
    a=ap.parse_args()
    out=(ROOT/a.out).resolve()
    if not out.is_relative_to(ROOT) or a.episodes<1: ap.error("Invalid output/count")
    out.mkdir(parents=True,exist_ok=False)
    env=OceanTrainingPool(ROOT/a.inventory);c=env.cfg
    rng=np.random.default_rng(a.seed)
    arrays={k:[] for k in ("context","target","found","received","priority","particles","weights","time_s","episode")}
    episodes=[];probe=encode_message(np.zeros(32))
    try:
        for ep in range(a.episodes):
            seed=int(rng.integers(0,2**31-1))
            if a.balanced:
                count=len(env.training_scenarios)
                selected=env.training_scenarios[ep%count]["id"]
                obs,_=env.reset_for_training_scenario(seed,selected)
                mode=(ep//count+ep%count)%len(CollectionPolicy.MODES)
            else:
                obs,_=env.reset(seed)
                mode=ep%len(CollectionPolicy.MODES)
            behavior=CollectionPolicy.MODES[mode]
            policy=CollectionPolicy(c,behavior,seed+100000)
            while not env.done:
                context=swarm_context(obs,c);target=target_slots(env)
                _,gradient=value_gradient(env,env.targets)
                gradient[env.found]=0
                unit=1/(2*np.pi*c.prior_sigma_m**2)
                priority=float(np.clip(np.linalg.norm(gradient*c.region_m/2/unit),.1,10))
                inbox=env.exchange_messages([probe]*c.n_uavs)
                received=np.zeros((c.n_uavs,c.n_uavs),dtype=bool)
                for receiver,items in enumerate(inbox):
                    for sender,_ in items: received[receiver,sender]=True
                row=dict(context=context,target=target,found=env.found.copy(),received=received,
                         priority=priority,particles=env.particles.copy(),weights=env.weights.copy(),
                         time_s=env.t,episode=ep)
                for key,value in row.items(): arrays[key].append(value)
                obs,_,_,_,info=env.step(policy.action(obs))
            episodes.append(dict(episode=ep,seed=seed,scenario_id=env.active_scenario_id,behavior=behavior,info=info))
            print(json.dumps(dict(episode=ep,rows=len(arrays["target"]))),flush=True)
        np.savez_compressed(out/"samples.npz",**{k:np.asarray(v) for k,v in arrays.items()})
        metadata=dict(context_schema=CONTEXT_SCHEMA,target_schema=TARGET_SCHEMA,split="train",
                      config=env.config_dict(),forcing=env.forcing_metadata(),args=vars(a),episodes=episodes,
                      samples_sha256=file_sha(out/"samples.npz"),source_sha256=file_sha(__file__),
                      particle_precision="float64 retained; independent beliefs, no future trajectory",
                      priority="provisional spatial gradient; recompute neural-decoder priority from snapshots",
                      limitation="identity-ordered targets; found flags separate; no permutation-invariant codec claim",
                      status="simulated training-only auxiliary collection; development inventory")
        (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    finally: env.close()

if __name__=="__main__": main()
