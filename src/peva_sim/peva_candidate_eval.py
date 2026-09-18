import argparse,json,time
from pathlib import Path
import numpy as np,torch
from peva_sim.env import Config,MaritimeSAR
from peva_sim.mappo_formal import Actor,feat

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--episodes',type=int,default=20); ap.add_argument('--particles',type=int,default=256); ap.add_argument('--seed',type=int,default=30000); ap.add_argument('--out',required=True); ap.add_argument('--guidance',type=float,default=.35); ap.add_argument('--packet-drop',type=float,default=.1); ap.add_argument('--forecast-bias',type=float,default=0.0); a=ap.parse_args()
 c=Config(particles=a.particles,packet_drop=a.packet_drop,forecast_bias_mps=a.forecast_bias); e=MaritimeSAR(c); o,_=e.reset(a.seed); d=feat(o,0,c,True).size; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ac=Actor(d).to(dev); ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); ac.load_state_dict(ck['actor']); ac.eval(); rows=[]; t0=time.perf_counter(); total_bytes=0
 for k in range(a.episodes):
  o,_=e.reset(a.seed+k); total=0.; steps=0
  while not e.done:
   x=torch.as_tensor(np.stack([feat(o,i,c,True) for i in range(c.n_uavs)]),device=dev)
   with torch.no_grad(): mu,_=ac(x)
   act=torch.tanh(mu).cpu().numpy()*c.speed_mps
   v=o['value_density']; grid=o['prior_grid_m']; best=grid[int(np.argmax(v))]
   for i in range(c.n_uavs):
    delta=best-o['positions_m'][i]; n=np.linalg.norm(delta)
    if n>1e-6: act[i]=(1-a.guidance)*act[i]+a.guidance*delta/n*c.speed_mps
   messages=[np.asarray(np.clip(act[i]/c.speed_mps*127,-127,127),np.int8).tobytes() for i in range(c.n_uavs)]
   total_bytes += sum(len(m)+c.header_bytes for m in messages)
   o,r,term,trunc,info=e.step(act,messages); total+=r; steps+=1
  rows.append({'seed':a.seed+k,'success_rate':info['success_rate'],'ttd_all':info['ttd_all'],'survival_score':info['survival_score'],'reward':total,'steps':steps})
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); result={'algorithm':'PEVA-Gen candidate: PVF guidance + quantized governance messages','guidance':a.guidance,'particles':a.particles,'episodes':rows,'mean_success_rate':float(np.mean([x['success_rate'] for x in rows])),'mean_ttd_all':float(np.mean([x['ttd_all'] for x in rows])),'mean_survival_score':float(np.mean([x['survival_score'] for x in rows])),'mean_peer_bytes_per_uav_step':float(total_bytes/(a.episodes*c.n_uavs*max(1,np.mean([x['steps'] for x in rows])))),'wall_s':time.perf_counter()-t0,'device':str(dev)}; out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
