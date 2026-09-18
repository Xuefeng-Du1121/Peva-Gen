import argparse,json,time
from pathlib import Path
import numpy as np,torch
from .real_dataset import RealDriftDataset
from .joint_modules import JointBeliefCommunication

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--out',required=True); a=ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(1); d=RealDriftDataset('runs/real-east-china-2018-01-utc-v2/segments.npz'); idx=d.split_by_track(seed=20260913)['train']; pos=d.positions_deg[idx]; uv=d.environment_uv_ms[idx];
 # Relative future location target; context is observation-compatible (current location, current flow and time).
 target=torch.tensor((pos[:,3]-pos[:,0])/np.array([18.,13.]),dtype=torch.float32); ctx=np.zeros((len(idx),16),np.float32); ctx[:,:2]=((pos[:,0]-np.array([117.5,19.5]))/np.array([18.,13.])).astype(np.float32); ctx[:,2:4]=uv[:,0].astype(np.float32); ctx[:,4]=np.linspace(0,1,len(idx)); target=torch.tensor(target); ctx=torch.tensor(ctx); m=JointBeliefCommunication(); opt=torch.optim.Adam(m.parameters(),3e-4); hist=[]; start=time.perf_counter()
 for step in range(a.steps):
  b=np.random.default_rng(a.seed+step).integers(len(idx),size=min(16,len(idx))); loss=m.loss(target[b],ctx[b]); opt.zero_grad(); loss['total'].backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step();
  if step%100==0: row={'step':step,'total':float(loss['total'].detach()),'belief':float(loss['belief'].detach()),'relevance':float(loss['relevance'].detach()),'rate':float(loss['rate'].detach())}; hist.append(row); print(json.dumps(row),flush=True)
 out=Path(a.out); out.mkdir(parents=True,exist_ok=False); torch.save({'model':m.state_dict(),'seed':a.seed,'data':'runs/real-east-china-2018-01-utc-v2/segments.npz','steps':a.steps},out/'checkpoint.pt'); (out/'metrics.json').write_text(json.dumps({'algorithm':'joint diffusion belief + CIB auxiliary','history':hist,'wall_s':time.perf_counter()-start},indent=2))
if __name__=='__main__': main()
