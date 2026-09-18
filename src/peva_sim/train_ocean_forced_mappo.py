"""Audited baseline path. Does not claim to implement PEVA-Gen."""
import argparse,json,random,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from torch import nn
from .env import Config,MaritimeSAR
from .ocean_forced_env import OceanForcedSAR
from .ocean_field import OceanField
from .protocol import file_sha
from .ppo_math import gae,disk_action,latent_log_prob

ROOT=Path(__file__).resolve().parents[2]

def features(obs,c,pvf):
    rows=[]
    for i in range(c.n_uavs):
        pos=obs["positions_m"][i]
        contacts=np.asarray(obs["contacts_m"][i])
        if len(contacts):
            contacts=contacts[np.argsort(np.linalg.norm(contacts-pos,axis=1))[:4]]
        slots=np.zeros((4,3),np.float32)
        for j,contact in enumerate(contacts):
            slots[j,:2]=(contact-pos)/c.region_m; slots[j,2]=1
        # Identical dimensions and parameter count across both baselines.
        field=np.asarray(obs["value_density"],np.float32)
        field=field/max(float(field.max()),1e-12) if pvf else np.zeros_like(field)
        rows.append(np.concatenate((pos/c.region_m,[obs["time_s"]/c.horizon_s],
                                    slots.ravel(),field)))
    return np.asarray(rows,dtype=np.float32)

def critic_state(e):
    s=e.state(); c=e.cfg
    return np.concatenate((s["uavs_m"].ravel()/c.region_m,
                           s["targets_m"].ravel()/c.region_m,
                           s["found"].astype(float),[s["time_s"]/c.horizon_s])).astype(np.float32)

class Model(nn.Module):
    def __init__(self,obs_dim,state_dim):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(obs_dim,128),nn.Tanh(),
                                 nn.Linear(128,128),nn.Tanh(),nn.Linear(128,2))
        self.logstd=nn.Parameter(torch.full((2,),-.5))
        self.critic=nn.Sequential(nn.Linear(state_dim,128),nn.Tanh(),
                                  nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,np.sqrt(2)); nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01)
        nn.init.orthogonal_(self.critic[-1].weight,1.)
    def policy(self,x):
        return self.actor(x),self.logstd.clamp(-4,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--updates",type=int,default=20)
    ap.add_argument("--rollout",type=int,default=256)
    ap.add_argument("--epochs",type=int,default=4)
    ap.add_argument("--gae-lambda",type=float,default=.95)
    ap.add_argument("--entropy-coef",type=float,default=0.)
    ap.add_argument("--minibatch",type=int,default=64)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--pvf",action="store_true")
    ap.add_argument("--guidance",type=float,default=0.0)
    ap.add_argument("--field",required=True)
    ap.add_argument("--origin-lon",type=float,required=True)
    ap.add_argument("--origin-lat",type=float,required=True)
    ap.add_argument("--start-utc",type=float,required=True)
    ap.add_argument("--n-targets",type=int,default=3)
    ap.add_argument("--peva-aux",default=None)
    ap.add_argument("--transport-aux",default=None)
    ap.add_argument("--coverage-mode",choices=("conditional","intensity"),default="intensity")
    ap.add_argument("--particles",type=int,default=2000)

    ap.add_argument("--snapshot-every",type=int,default=50)

    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    if not 0 <= a.gae_lambda <= 1:
        ap.error("GAE lambda must be in [0,1]")
    if not np.isfinite(a.entropy_coef) or a.entropy_coef < 0:
        ap.error("Entropy coefficient must be finite and nonnegative")
    if a.transport_aux or a.peva_aux:
        ap.error("This trainer is an ocean-forced baseline; incompatible recorded-track auxiliaries forbidden")
    if a.guidance != 0:
        ap.error("External action guidance is not PEVA-Gen; use baseline without guidance")
    if min(a.updates,a.rollout,a.epochs,a.minibatch,a.particles,a.snapshot_every)<1:
        ap.error("Training counts must be positive")
    out=(ROOT/a.out).resolve()
    if not out.is_relative_to(ROOT): ap.error("Output must stay under project")
    out.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c=replace(Config(),particles=a.particles,n_targets=a.n_targets,windage=0.,coverage_mode=a.coverage_mode)
    field_path=(ROOT/a.field).resolve()
    if not field_path.is_relative_to(ROOT): ap.error("Field must stay under project")
    e=OceanForcedSAR(OceanField(field_path),[a.origin_lon,a.origin_lat],a.start_utc,c)
    episode_rng=np.random.default_rng(a.seed+90000)
    def reset():
        return e.reset(int(episode_rng.integers(0,2**31-1)))[0]
    obs=reset()
    adapter=None
    if a.transport_aux:
        from .transport_features import TransportFeatureAdapter
        adapter=TransportFeatureAdapter(ROOT/a.transport_aux,c,a.seed+123000)
        feature_fn=lambda z: adapter.step(e,z)
    elif a.peva_aux:
        from .peva_features import PEVAFeatureExtractor
        feature_fn=PEVAFeatureExtractor(a.peva_aux,c,a.pvf)
    else:
        feature_fn=lambda z: features(z,c,a.pvf)
    model=Model(adapter.feature_dim if adapter else feature_fn(obs).shape[-1],len(critic_state(e))).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5)
    metadata={"version":"mappo_ocean_forced_v1","status":"historical-forcing simulated training; single-patch development, not formal protocol",
              "args":vars(a),"config":e.config_dict(),"device":str(device),
              "parameters":sum(p.numel() for p in model.parameters()),
              "gamma":.99,
              "exploration_regularizer":"latent Gaussian entropy; not transformed-action entropy",
              "field_sha256":file_sha(field_path),"forcing":e.forcing_metadata(),
              "trajectory_source":"online stochastic simulation; recorded trajectories are not loaded",
              "source_sha256":file_sha(__file__),
              "guidance":a.guidance,
              "actor_inputs":"own position,time,local contacts; PVF channel or matched zero channel",
              "critic_inputs":"privileged global state; training only",
              "action_distribution":"Normal latent with bijective disk transform",
              "mission_timeout":"finite-horizon terminal, zero bootstrap"}
    if adapter:
        metadata.update(version="mappo_frozen_transport_aux_v1",
                        status="frozen auxiliary integration pilot; NOT complete PEVA-Gen",
                        auxiliary_sha256=adapter.checkpoint_sha256,
                        auxiliary_metadata=adapter.metadata,
                        actor_inputs="own position,time,contacts; received message context; belief mean and spread",
                        communication="pre-action 42-byte packets plus radio headers; uniform aggregation",
                        auxiliary_source_sha256=file_sha(ROOT/"src/peva_sim/transport_features.py"))
    from .reproducibility import snapshot_source
    metadata["source_archive_sha256"]=snapshot_source(out)
    metadata["checkpoint_scope"]="inference/learning-curve checkpoints; exact environment resume not implemented"
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    def tensor(x): return torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
    completed=[]; episode_return=0.; total_steps=0; start=time.perf_counter()
    for update in range(a.updates):
        buf={k:[] for k in ("x","s","raw","lp","v","nv","r","boot","cont")}
        for _ in range(a.rollout):
            x=feature_fn(obs); s=critic_state(e)
            with torch.no_grad():
                mu,ls=model.policy(tensor(x))
                raw=mu+ls.exp()*torch.randn_like(mu)
                lp=latent_log_prob(mu,ls,raw)
                action=disk_action(raw,c.speed_mps).cpu().numpy()
                value=model.critic(tensor(s)).item()
            nxt,reward,term,trunc,info=e.step(action)
            ended=term or trunc
            with torch.no_grad():
                nv=0. if ended else model.critic(tensor(critic_state(e))).item()
            for k,v in dict(x=x,s=s,raw=raw.cpu().numpy(),lp=lp.cpu().numpy(),
                            v=value,nv=nv,r=reward,boot=float(not ended),
                            cont=float(not ended)).items():
                buf[k].append(v)
            episode_return+=reward; total_steps+=1; obs=nxt
            if ended:
                completed.append(dict(info,return_value=episode_return,environment_step=total_steps))
                episode_return=0.; obs=reset()
        adv,returns=gae(buf["r"],buf["v"],buf["nv"],buf["boot"],buf["cont"],gamma=.99,lam=a.gae_lambda)
        x=tensor(buf["x"]); state=tensor(buf["s"]); raw=tensor(buf["raw"]); oldlp=tensor(buf["lp"])
        advantage=tensor((adv-adv.mean())/(adv.std()+1e-8))
        targets=tensor(returns)
        with torch.no_grad():
            mu,ls=model.policy(x)
            before=torch.exp(latent_log_prob(mu,ls,raw)-oldlp)
            ratio_error=float((before-1).abs().max().item())
        if ratio_error>1e-5: raise RuntimeError("Stored behavior probability inconsistent")
        losses=[]
        for epoch in range(a.epochs):
            for idx in torch.randperm(a.rollout,device=device).split(a.minibatch):
                mu,ls=model.policy(x[idx])
                log_ratio=latent_log_prob(mu,ls,raw[idx])-oldlp[idx]
                ratio=log_ratio.exp()
                aa=advantage[idx,None]
                actor_loss=-torch.minimum(ratio*aa,ratio.clamp(.8,1.2)*aa).mean()
                value_loss=.5*(model.critic(state[idx]).squeeze(-1)-targets[idx]).square().mean()
                latent_entropy=torch.distributions.Normal(mu,ls.exp()).entropy().sum(-1).mean()
                loss=actor_loss+value_loss-a.entropy_coef*latent_entropy
                if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
                opt.zero_grad(); loss.backward()
                grad=nn.utils.clip_grad_norm_(model.parameters(),.5)
                if not torch.isfinite(grad): raise RuntimeError("Non-finite gradient")
                opt.step()
                losses.append(float(loss.detach().item()))
        row={"update":update,"environment_steps":total_steps,"completed_episodes":len(completed),
             "current_episode_time_s":e.t,"rollout_reward":sum(buf["r"]),
             "initial_ratio_max_error":ratio_error,"mean_loss":float(np.mean(losses)),
             "elapsed_s":time.perf_counter()-start}
        with (out/"updates.jsonl").open("a") as f: f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
        payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"metadata":metadata,
                    "auxiliary_model":adapter.model.state_dict() if adapter else None,
                    "auxiliary_rng":adapter.rng.get_state() if adapter else None,
                    "update":update,"environment_steps":total_steps}
        torch.save(payload,out/"checkpoint.pt")
        if (update+1)%a.snapshot_every==0 or update+1==a.updates:
            torch.save(payload,out/f"checkpoint-step-{total_steps:09d}.pt")
            (out/"episodes.json").write_text(json.dumps(completed,indent=2))
    (out/"episodes.json").write_text(json.dumps(completed,indent=2))
    if hasattr(e,"close"): e.close()
if __name__=="__main__": main()
