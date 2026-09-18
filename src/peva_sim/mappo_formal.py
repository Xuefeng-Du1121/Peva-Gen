import argparse,json,random,time
from pathlib import Path
import numpy as np, torch
from torch import nn
from peva_sim.env import Config,MaritimeSAR
from peva_sim.real_env import RealMaritimeSAR

class Actor(nn.Module):
 def __init__(self,d):
  super().__init__(); self.body=nn.Sequential(nn.Linear(d,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh()); self.mu=nn.Linear(128,2); self.logstd=nn.Parameter(torch.zeros(2))
 def forward(self,x): return self.mu(self.body(x)),self.logstd.clamp(-4,1)
class Critic(nn.Module):
 def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.Tanh(),nn.Linear(256,128),nn.Tanh(),nn.Linear(128,1))
 def forward(self,x): return self.net(x).squeeze(-1)
def feat(o,i,c,pvf=False):
 p=(o["positions_m"][i]/c.region_m).astype(np.float32); g=(o["prior_grid_m"]/c.region_m).reshape(-1).astype(np.float32); v=o["value_density"]; v=(v/(v.max()+1e-12)).astype(np.float32)
 return np.concatenate((p,np.array([o["time_s"]/c.horizon_s],np.float32),g,v if pvf else np.empty(0,np.float32)))
def logprob(dist,raw): return dist.log_prob(raw).sum(-1)-torch.log(1-torch.tanh(raw).pow(2)+1e-6).sum(-1)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--updates",type=int,default=16); ap.add_argument("--rollout",type=int,default=128); ap.add_argument("--epochs",type=int,default=4); ap.add_argument("--minibatch",type=int,default=256); ap.add_argument("--seed",type=int,default=0); ap.add_argument("--pvf",action="store_true"); ap.add_argument("--real-data",action="store_true"); ap.add_argument("--guidance",type=float,default=0.0); ap.add_argument("--out",default="runs/mappo-formal-001"); a=ap.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); c=Config();
 if a.real_data: c=__import__("dataclasses").replace(c,n_targets=1)
 e=RealMaritimeSAR("runs/real-east-china-2018-01-utc-v2/segments.npz",c,split="train",seed=a.seed) if a.real_data else MaritimeSAR(c); o,_=e.reset(a.seed); d=feat(o,0,c,a.pvf).size; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); ac=Actor(d).to(dev); cr=Critic(d*c.n_uavs).to(dev); opta=torch.optim.Adam(ac.parameters(),3e-4); optc=torch.optim.Adam(cr.parameters(),3e-4); history=[]; out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
 for u in range(a.updates):
  e.reset(a.seed+u); obs,_=e.reset(a.seed+u); X=[]; A=[]; LP=[]; V=[]; R=[]; M=[]; ret=0.; done=False
  for t in range(a.rollout):
   x=np.stack([feat(obs,i,c,a.pvf) for i in range(c.n_uavs)]); xt=torch.as_tensor(x,device=dev);
   with torch.no_grad():
    mu,ls=ac(xt); dist=torch.distributions.Normal(mu,ls.exp()); raw=dist.sample(); act=torch.tanh(raw).cpu().numpy()*c.speed_mps
   if a.guidance>0 and a.pvf:
    best=obs["prior_grid_m"][int(np.argmax(obs["value_density"]))]
    for ii in range(c.n_uavs):
     dd=best-obs["positions_m"][ii]; norm=np.linalg.norm(dd)
     if norm>1e-6: act[ii]=(1-a.guidance)*act[ii]+a.guidance*dd/norm*c.speed_mps
   if a.guidance>0 and a.pvf:
    best=obs['prior_grid_m'][int(np.argmax(obs['value_density']))]
    for ii in range(c.n_uavs):
     dd=best-obs['positions_m'][ii]; norm=np.linalg.norm(dd)
     if norm>1e-6: act[ii]=(1-a.guidance)*act[ii]+a.guidance*dd/norm*c.speed_mps
   guided_raw=torch.atanh(torch.as_tensor((act/c.speed_mps).clip(-.999,.999),device=dev)); lp=logprob(dist,guided_raw).sum(); val=cr(xt.reshape(1,-1))
   no,r,term,trunc,info=e.step(act); X.append(x); A.append(np.asarray(act)); LP.append(lp.item()); V.append(val.item()); R.append(r); M.append(0. if term else 1.); ret+=r; obs=no
   if term or trunc: obs,_=e.reset(a.seed+u+1000+t); done=True
  with torch.no_grad():
   nx=torch.as_tensor(np.stack([feat(obs,i,c,a.pvf) for i in range(c.n_uavs)]),device=dev); nv=cr(nx.reshape(1,-1)).item()
  adv=np.zeros(len(R),np.float32); gae=0.
  for t in reversed(range(len(R))):
   delta=R[t]+.99*(nv if t==len(R)-1 else V[t+1])*M[t]-V[t]; gae=delta+.99*.95*M[t]*gae; adv[t]=gae
  returns=adv+np.asarray(V,np.float32); xs=torch.as_tensor(np.asarray(X),device=dev); acts=torch.as_tensor(np.asarray(A),device=dev); oldlp=torch.as_tensor(LP,device=dev); glob=xs.reshape(len(xs),-1); advt=torch.as_tensor((adv-adv.mean())/(adv.std()+1e-8),device=dev); rett=torch.as_tensor(returns,device=dev)
  n=len(xs)
  for _ in range(a.epochs):
   perm=torch.randperm(n,device=dev)
   for ix in perm.split(a.minibatch):
    mu,ls=ac(xs[ix].reshape(-1,d)); dist=torch.distributions.Normal(mu,ls.exp()); raw=torch.atanh((acts[ix].reshape(-1,2)/c.speed_mps).clamp(-.999,.999)); nlp=logprob(dist,raw).reshape(len(ix),c.n_uavs).sum(-1); ratio=torch.exp(nlp-oldlp[ix]); pl=-torch.minimum(ratio*advt[ix],ratio.clamp(.8,1.2)*advt[ix]).mean(); ent=dist.entropy().sum(-1).mean(); loss=pl-.001*ent; opta.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(),.5); opta.step()
    vl=((cr(glob)-rett)**2).mean(); optc.zero_grad(); vl.backward(); nn.utils.clip_grad_norm_(cr.parameters(),.5); optc.step()
  row={"update":u,"return":ret,"policy_loss":float(pl.detach()),"value_loss":float(vl.detach()),"success_rate":info["success_rate"],"episodes_ended":int(done),"device":str(dev),"pvf":a.pvf,"guidance":a.guidance,"real_data":a.real_data}; history.append(row); print(json.dumps(row),flush=True)
 torch.save({"actor":ac.state_dict(),"critic":cr.state_dict(),"config":e.config_dict(),"seed":a.seed,"pvf":a.pvf},out/"checkpoint.pt"); (out/"metrics.json").write_text(json.dumps({"algorithm":"MAPPO-CTDE-formal","history":history,"seed":a.seed,"pvf":a.pvf,"device":str(dev)},indent=2))
if __name__=="__main__": main()
