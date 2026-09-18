import argparse,json,time
from pathlib import Path
import numpy as np,torch
from .real_dataset import RealDriftDataset
from .diffusion_belief import DiffusionBelief
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=100); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--out',required=True); a=ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); d=RealDriftDataset('runs/real-east-china-2018-01-utc-v2/segments.npz'); idx=d.split_by_track(seed=20260913)['train']; pos=d.positions_deg[idx]; uv=d.environment_uv_ms[idx]; target=torch.as_tensor((pos[:,3]-pos[:,0])/np.array([18.,13.]),dtype=torch.float32); ctx=np.zeros((len(idx),16),np.float32); ctx[:,:2]=((pos[:,0]-[117.5,19.5])/[18.,13.]).astype(np.float32); ctx[:,2:4]=uv[:,0]; ctx[:,4]=np.linspace(0,1,len(idx)); ctx=torch.as_tensor(ctx); m=DiffusionBelief(); opt=torch.optim.Adam(m.parameters(),3e-4); rng=np.random.default_rng(a.seed); hist=[]; start=time.perf_counter(); torch.set_num_threads(1)
 for step in range(a.steps):
  b=torch.as_tensor(rng.integers(len(idx),size=min(64,len(idx))),dtype=torch.long); loss=m.training_loss(target[b],ctx[b]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step();
  if step%10==0: hist.append({'step':step,'loss':float(loss.detach())}); print(hist[-1],flush=True)
 out=Path(a.out); out.mkdir(parents=True,exist_ok=False); torch.save({'model':m.state_dict(),'seed':a.seed,'data_sha256':__import__('hashlib').sha256(Path('runs/real-east-china-2018-01-utc-v2/segments.npz').read_bytes()).hexdigest()},out/'checkpoint.pt'); (out/'metrics.json').write_text(json.dumps({'history':hist,'wall_s':time.perf_counter()-start},indent=2)); print('saved',out)
if __name__=='__main__': main()
