import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from peva_sim.env import Config, MaritimeSAR

class Actor(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,2)); self.logstd=nn.Parameter(torch.zeros(2))
    def forward(self,x): return self.net(x), self.logstd.clamp(-4,1)

class Critic(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def feature(obs,i,cfg):
    p=(obs["positions_m"][i]/cfg.region_m).astype(np.float32)
    grid=(obs["prior_grid_m"]/cfg.region_m).astype(np.float32).reshape(-1)
    val=(obs["value_density"]/max(float(obs["value_density"].max()),1e-12)).astype(np.float32)
    q=np.zeros(8,np.float32)
    ct=obs["contacts_m"][i][:4].reshape(-1)[:8]/cfg.region_m
    q[:len(ct)]=ct
    return np.concatenate((p, np.array([obs["time_s"]/cfg.horizon_s],np.float32),grid,val,q))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--updates",type=int,default=8); ap.add_argument("--rollout",type=int,default=128); ap.add_argument("--seed",type=int,default=0); ap.add_argument("--out",default="runs/mappo-smoke-001"); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root=Path(__file__).resolve().parents[2]; out=(root/args.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    c=Config(); env=MaritimeSAR(c); obs,_=env.reset(args.seed)
    d=feature(obs,0,c).size; device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor=Actor(d).to(device); critic=Critic(d*c.n_uavs).to(device)
    ao=torch.optim.Adam(actor.parameters(),3e-4); co=torch.optim.Adam(critic.parameters(),3e-4)
    log=[]; buf=[]
    for update in range(args.updates):
        obs,_=env.reset(args.seed+update); episode_reward=0.
        for t in range(args.rollout):
            x=np.stack([feature(obs,i,c) for i in range(c.n_uavs)]); xt=torch.tensor(x,device=device)
            with torch.no_grad():
                mu,ls=actor(xt); dist=torch.distributions.Normal(mu,ls.exp()); z=dist.sample(); act=torch.tanh(z)*c.speed_mps
                lp=dist.log_prob(z).sum(-1)-torch.log(1-torch.tanh(z).pow(2)+1e-6).sum(-1)
                v=critic(xt.reshape(1,-1))
            nxt,r,term,trunc,info=env.step(act.cpu().numpy())
            buf.append((x,act.cpu().numpy(),lp.sum().item(),v.item(),r))
            episode_reward+=r; obs=nxt
            if term or trunc: obs,_=env.reset(args.seed+update+1000); break
        R=0.; returns=[]
        for *_,r in reversed(buf[-args.rollout:]): R=r+0.99*R; returns.append(R)
        returns=torch.tensor(returns[::-1],device=device); xs=torch.tensor(np.stack([b[0] for b in buf[-len(returns):]]),device=device)
        acts=torch.tensor(np.stack([b[1] for b in buf[-len(returns):]]),device=device)
        oldlp=torch.tensor([b[2] for b in buf[-len(returns):]],device=device); glob=xs.reshape(len(xs),-1)
        adv=(returns-critic(glob)).detach(); adv=(adv-adv.mean())/(adv.std()+1e-8)
        mu,ls=actor(xs.reshape(-1,d)); dist=torch.distributions.Normal(mu,ls.exp()); raw=torch.atanh(acts.reshape(-1,2).clamp(-.999,.999)/c.speed_mps)
        nlp=(dist.log_prob(raw).sum(-1)-torch.log(1-torch.tanh(raw).pow(2)+1e-6).sum(-1)).reshape(len(xs),c.n_uavs).sum(-1)
        ratio=torch.exp(nlp-oldlp); loss=-(torch.min(ratio*adv,(ratio.clamp(.8,1.2))*adv)).mean()
        ao.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(),0.5); ao.step()
        vl=((critic(glob)-returns)**2).mean(); co.zero_grad(); vl.backward(); nn.utils.clip_grad_norm_(critic.parameters(),0.5); co.step()
        row={"update":update,"reward":episode_reward,"policy_loss":float(loss),"value_loss":float(vl),"device":str(device),"success_rate":info["success_rate"]}
        log.append(row); print(json.dumps(row),flush=True)
    torch.save({"actor":actor.state_dict(),"critic":critic.state_dict(),"config":env.config_dict(),"seed":args.seed},out/"checkpoint.pt")
    (out/"metrics.json").write_text(json.dumps({"algorithm":"minimal MAPPO CTDE","note":"smoke baseline only","metrics":log},indent=2))
if __name__=="__main__": main()
