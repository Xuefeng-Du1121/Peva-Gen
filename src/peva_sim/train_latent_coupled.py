"""Interleaved real-environment coupled training pilot; not yet full PEVA-Gen.

Remaining: value-weighted transmitted relevance and manuscript latent decoder.
Implements fresh on-policy collection then a common value-prioritized update.
"""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from .env import Config
from .real_env import RealMaritimeSAR
from .mappo_v2 import Model,critic_state
from .latent_features import LatentFeatureAdapter as TransportFeatureAdapter
from .belief_contract import swarm_context,training_target
from .physics_priority import value_gradient
from .ppo_math import disk_action,latent_log_prob,gae
from .latent_coupled_update import LatentCoupledUpdater as CoupledUpdater
from .protocol import file_sha
ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--transport-aux",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--message-weighting",choices=("uniform","physics_grid"),default="uniform")
    ap.add_argument("--updates",type=int,default=10);ap.add_argument("--rollout",type=int,default=256)
    ap.add_argument("--batch",type=int,default=256);ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()
    if min(a.updates,a.rollout,a.batch)<1: ap.error("positive counts required")
    out=(ROOT/a.out).resolve();auxpath=(ROOT/a.transport_aux).resolve()
    if not out.is_relative_to(ROOT) or not auxpath.is_relative_to(ROOT): ap.error("paths outside project")
    c=Config(n_targets=1,coverage_mode="intensity")
    adapter=TransportFeatureAdapter(auxpath,c,a.seed+123000,message_weighting=a.message_weighting)
    if adapter.metadata.get("latent_dim")!=128:
        ap.error("128-dimensional auxiliary required")
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=ROOT/"runs/real-east-china-2018-01-utc-v2/segments.npz";manifest=data.parent/"split-manifest.json"
    env=RealMaritimeSAR(data,c,split="train",manifest_path=manifest)
    episode_rng=np.random.default_rng(a.seed+90000)
    def reset():
        return env.reset(int(episode_rng.integers(0,2**31-1)))[0]
    obs=reset()
    policy=Model(adapter.feature_dim,len(critic_state(env)))
    for parameter in adapter.model.parameters(): parameter.requires_grad_(True)
    adapter.model.codec.freeze()
    popt=torch.optim.Adam(policy.parameters(),lr=3e-4,eps=1e-5)
    aopt=torch.optim.Adam([p for p in adapter.model.parameters() if p.requires_grad],lr=3e-4)
    updater=CoupledUpdater(policy,adapter.model,popt,aopt,c.n_uavs,seed=a.seed+345000)
    out.mkdir(parents=True,exist_ok=False)
    args={**vars(a),"pvf":False,"guidance":0,"data":str(data),"manifest":str(manifest)}
    metadata={"version":"coupled_latent128_pilot_v1","status":"128-D coupled development; not complete PEVA-Gen",
              "args":args,"config":env.config_dict(),"device":"cpu",
              "data_sha256":file_sha(data),"manifest_sha256":file_sha(manifest),
              "auxiliary_sha256":file_sha(auxpath),"auxiliary_metadata":adapter.metadata,
              "communication":"pre-action 42-byte packets; actual radio mask",
              "message_weighting":a.message_weighting,
              "replay":"fresh rollout only; frozen priority distribution per update",
              "uncorrected_terms":"belief and CIB relevance",
              "corrected_terms":"CIB rate and critic: exact 1/(N*p); actor: clipped correction times risk gate",
              "fusion":"one global beta/update; alignment and normalized TD EMA; ensemble omitted",
              "latent":"128-D frozen learned target codec; exact neural decoder gradient pullback",
              "alignment_weight":.1,"actor_correction_clip":[.2,5],"belief_samples":2,
              "protocol_limit":"recorded train tracks; not simulated-drift training",
              "sources":{p.name:file_sha(p) for p in sorted((ROOT/"src/peva_sim").glob("*.py"))}}
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    episodes=[];total_steps=0;episode_return=0.;started=time.perf_counter()
    def tensor(values): return torch.as_tensor(np.asarray(values),dtype=torch.float32)
    try:
        for update in range(a.updates):
            buf={k:[] for k in ("state","features","raw","old_log_prob","physics_gradient","context",
                               "target","score","received","spread","reward","value","next_value","boot","cont")}
            adapter.model.eval()
            for _ in range(a.rollout):
                context=swarm_context(obs,c);target=training_target(env.state(),c)
                with torch.no_grad():
                    target_z=adapter.model.codec.encode_target(torch.tensor(target,dtype=torch.float32)[None])
                    decoded_xy=adapter.model.codec.decode_locations_m(target_z,c.region_m)
                _,gx=value_gradient(env,decoded_xy.numpy().reshape(-1,2))
                density_unit=1/(2*np.pi*c.prior_sigma_m**2)
                gx=torch.tensor(gx[None]/density_unit,dtype=torch.float32)
                pg=adapter.model.codec.pullback_location_gradient(target_z,gx,c.region_m)[0].numpy()
                state=critic_state(env)
                features=adapter.step(env,obs)
                received=np.zeros((c.n_uavs,c.n_uavs),dtype=bool)
                for receiver,items in enumerate(env.inbox):
                    for sender,_ in items: received[receiver,sender]=True
                with torch.no_grad():
                    mu,ls=policy.policy(tensor(features))
                    raw=mu+ls.exp()*torch.randn_like(mu)
                    lp=latent_log_prob(mu,ls,raw)
                    action=disk_action(raw,c.speed_mps).numpy()
                    value=float(policy.critic(tensor(state)))
                nxt,reward,term,trunc,info=env.step(action);ended=term or trunc
                with torch.no_grad():
                    nv=0. if ended else float(policy.critic(tensor(critic_state(env))))
                row=dict(state=state,features=features,raw=raw.numpy(),old_log_prob=lp.numpy(),
                         physics_gradient=pg,context=context,target=target,score=adapter.last["message_scores"].copy(),
                         received=received,spread=adapter.last["spread"].copy(),reward=reward,
                         value=value,next_value=nv,boot=float(not ended),cont=float(not ended))
                for key,value in row.items(): buf[key].append(value)
                total_steps+=1;episode_return+=reward;obs=nxt
                if ended:
                    episodes.append(dict(info,return_value=episode_return,environment_step=total_steps))
                    episode_return=0.;obs=reset()
            adv,ret=gae(buf["reward"],buf["value"],buf["next_value"],buf["boot"],buf["cont"])
            rollout={k:tensor(v) for k,v in buf.items() if k not in ("received",)}
            rollout["received"]=torch.tensor(np.asarray(buf["received"]),dtype=torch.bool)
            rollout["advantage"]=tensor((adv-adv.mean())/(adv.std()+1e-8))
            rollout["returns"]=tensor(ret)
            rollout["td_error"]=tensor(np.asarray(buf["reward"])+.99*np.asarray(buf["boot"])*np.asarray(buf["next_value"])-np.asarray(buf["value"]))
            metrics=updater.update(rollout,a.batch)
            metrics.update(update=update,environment_steps=total_steps,completed_episodes=len(episodes),
                           elapsed_s=time.perf_counter()-started)
            with (out/"updates.jsonl").open("a") as f: f.write(json.dumps(metrics)+"\n")
            print(json.dumps(metrics),flush=True)
            torch.save({"model":policy.state_dict(),"optimizer":popt.state_dict(),
                        "auxiliary_model":adapter.model.state_dict(),"auxiliary_optimizer":aopt.state_dict(),
                        "auxiliary_rng":adapter.rng.get_state(),"torch_rng":torch.get_rng_state(),
                        "sampling_rng":updater.rng.bit_generator.state,"ema_td":updater.ema_td,
                        "metadata":metadata,"update":update,"environment_steps":total_steps},out/"checkpoint.pt")
        (out/"episodes.json").write_text(json.dumps(episodes,indent=2))
    finally: env.close()
if __name__=="__main__": main()
