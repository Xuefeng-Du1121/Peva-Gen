"""Collect current-state belief warm-start data; never reads held-out labels."""
import argparse,json
from pathlib import Path
import numpy as np
from .env import Config
from .real_env import RealMaritimeSAR
from .belief_contract import swarm_context,training_target,CONTEXT_SCHEMA,TARGET_SCHEMA
from .communication import encode_message
from .protocol import file_sha
from .physics_priority import normalized_priority,PRIORITY_SCHEMA
from .collection_policy import CollectionPolicy

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--episodes",type=int,default=2)
    ap.add_argument("--coverage-mode",choices=("conditional","intensity"),default="intensity")
    ap.add_argument("--split",choices=("train","validation"),default="train")
    ap.add_argument("--behavior",choices=("pvf","mixed"),default="mixed")
    ap.add_argument("--particles",type=int,default=2000)
    ap.add_argument("--seed",type=int,default=71)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    if a.episodes<1 or a.particles<1: ap.error("positive counts required")
    out=(ROOT/a.out).resolve()
    if not out.is_relative_to(ROOT): ap.error("output outside project")
    out.mkdir(parents=True,exist_ok=False)
    data=ROOT/"runs/real-east-china-2018-01-utc-v2/segments.npz"
    manifest=data.parent/"split-manifest.json"
    c=Config(n_targets=1,particles=a.particles,coverage_mode=a.coverage_mode)
    e=RealMaritimeSAR(data,c,split=a.split,manifest_path=manifest)
    rng=np.random.default_rng(a.seed)
    arrays={k:[] for k in ("context","target","received","priority","episode","time_s","global_index")}
    episodes=[]
    probe=encode_message(np.zeros(32))
    try:
        for ep in range(a.episodes):
            seed=int(rng.integers(0,2**31-1))
            index=int(rng.integers(len(e.replay.indices)))
            obs,_=e.reset(seed,episode_index=index)
            behavior="pvf" if a.behavior=="pvf" else CollectionPolicy.MODES[ep%3]
            controller=CollectionPolicy(c,behavior,seed+100000)
            steps=0
            while True:
                context=swarm_context(obs,c)
                target=training_target(e.state(),c)
                priority=normalized_priority(e,e.targets,obs)
                inbox=e.exchange_messages([probe]*c.n_uavs)
                received=np.zeros((c.n_uavs,c.n_uavs),dtype=bool)
                for receiver,items in enumerate(inbox):
                    for sender,_ in items: received[receiver,sender]=True
                for key,value in dict(context=context,target=target,received=received,priority=priority,
                                      episode=ep,time_s=e.t,global_index=e.replay.indices[index]).items():
                    arrays[key].append(value)
                action=controller.action(obs)
                obs,_,term,trunc,info=e.step(action); steps+=1
                if term or trunc: break
            episodes.append(dict(episode=ep,seed=seed,global_index=int(e.replay.indices[index]),
                                 track_id=str(e.episode["track_id"]),start_utc=e.start_utc,
                                 steps=steps,behavior=behavior,info=info))
            print(json.dumps({"episode":ep,"rows":len(arrays["target"]),"steps":steps}),flush=True)
    finally:
        e.close()
    np.savez_compressed(out/"samples.npz",**{k:np.asarray(v) for k,v in arrays.items()})
    metadata={"context_schema":CONTEXT_SCHEMA,"target_schema":TARGET_SCHEMA,
              "data_sha256":file_sha(data),"manifest_sha256":file_sha(manifest),
              "samples_sha256":file_sha(out/"samples.npz"),
              "source_sha256":file_sha(__file__),"split":a.split,"config":e.config_dict(),
              "priority_schema":PRIORITY_SCHEMA,
              "args":vars(a),"episodes":episodes,
              "communication":"pre-action zero-content 42-byte probe packets to sample link masks",
              "status":"warm-start sample collection only; no learned communication claim",
              "dependencies":{p:file_sha(ROOT/"src/peva_sim"/p) for p in
                              ("env.py","real_env.py","belief_contract.py","communication.py","physics_priority.py","collection_policy.py")}}
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps({"complete":True,"samples":len(arrays["target"]),"out":str(out)}),flush=True)


if __name__=="__main__": main()
