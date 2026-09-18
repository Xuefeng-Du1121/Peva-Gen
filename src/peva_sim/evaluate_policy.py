import argparse,json,time
from pathlib import Path
import numpy as np, torch
from peva_sim.env import Config,MaritimeSAR
from peva_sim.mappo_formal import Actor,feat

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--episodes',type=int,default=100); ap.add_argument('--particles',type=int,default=2000); ap.add_argument('--pvf',action='store_true'); ap.add_argument('--seed',type=int,default=10000); ap.add_argument('--out',required=True); a=ap.parse_args()
 c=Config(particles=a.particles); e=MaritimeSAR(c); o,_=e.reset(a.seed); d=feat(o,0,c,a.pvf).size; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ac=Actor(d).to(dev); ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); ac.load_state_dict(ck['actor']); ac.eval(); rows=[]; t0=time.perf_counter()
 for k in range(a.episodes):
  o,_=e.reset(a.seed+k); total=0.; steps=0
  while not e.done:
   x=torch.as_tensor(np.stack([feat(o,i,c,a.pvf) for i in range(c.n_uavs)]),device=dev)
   with torch.no_grad(): mu,_=ac(x); act=torch.tanh(mu)*c.speed_mps
   o,r,term,trunc,info=e.step(act.cpu().numpy()); total+=r; steps+=1
  rows.append({'seed':a.seed+k,'success_rate':info['success_rate'],'ttd_all':info['ttd_all'],'survival_score':info['survival_score'],'reward':total,'steps':steps,'bytes':info['bytes']})
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); result={'checkpoint':a.checkpoint,'episodes':rows,'mean_success_rate':float(np.mean([x['success_rate'] for x in rows])),'mean_ttd_all':float(np.mean([x['ttd_all'] for x in rows])),'mean_survival_score':float(np.mean([x['survival_score'] for x in rows])),'wall_s':time.perf_counter()-t0,'device':str(dev),'pvf':a.pvf}; out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
