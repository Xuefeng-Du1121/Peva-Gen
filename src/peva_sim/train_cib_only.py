import argparse,json,time
from pathlib import Path
import numpy as np,torch
from .real_dataset import RealDriftDataset
from .peva_gen import ValueAlignedBottleneck
from .joint_modules import CIBDecoder
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=500); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--out',required=True); a=ap.parse_args(); torch.manual_seed(a.seed); d=RealDriftDataset('runs/real-east-china-2018-01-utc-v2/segments.npz'); idx=d.split_by_track(seed=20260913)['train']; pos=d.positions_deg[idx]; uv=d.environment_uv_ms[idx]; target=torch.as_tensor((pos[:,3]-pos[:,0])/[18.,13.],dtype=torch.float32); ctx=np.zeros((len(idx),16),np.float32); ctx[:,:2]=((pos[:,0]-[117.5,19.5])/[18.,13.]).astype(np.float32); ctx[:,2:4]=uv[:,0]; ctx[:,4]=np.linspace(0,1,len(idx)); ctx=torch.as_tensor(ctx); enc=ValueAlignedBottleneck(16,32); dec=CIBDecoder(32,2); opt=torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),3e-4); rng=np.random.default_rng(a.seed); hist=[]; start=time.perf_counter(); torch.set_num_threads(1)
 for step in range(a.steps):
  b=torch.as_tensor(rng.integers(len(idx),size=min(128,len(idx))),dtype=torch.long); msg,kl=enc(ctx[b]); recon=(dec(msg)-target[b]).square().mean(); loss=recon+1e-3*kl.mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),1); opt.step()
  if step%50==0: hist.append({'step':step,'loss':float(loss.detach()),'relevance':float(recon.detach()),'kl':float(kl.mean().detach())}); print(hist[-1],flush=True)
 out=Path(a.out); out.mkdir(parents=True,exist_ok=False); torch.save({'encoder':enc.state_dict(),'decoder':dec.state_dict(),'seed':a.seed},out/'checkpoint.pt'); (out/'metrics.json').write_text(json.dumps({'history':hist,'wall_s':time.perf_counter()-start},indent=2)); print('saved',out)
if __name__=='__main__': main()
