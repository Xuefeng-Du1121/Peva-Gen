import argparse,json,time
from pathlib import Path
import numpy as np,torch
from .env import Config
from .real_env import RealMaritimeSAR
from .mappo_v2 import Model,features,critic_state,ROOT

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--episodes',type=int,default=20); ap.add_argument('--seed',type=int,default=95000); ap.add_argument('--guidance',type=float,default=.7); ap.add_argument('--out',required=True); a=ap.parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); meta=ck['metadata']; c=Config(**{k:v for k,v in meta['config'].items() if k in Config.__dataclass_fields__}); e=RealMaritimeSAR(ROOT/meta['args']['data'],c,split='test',seed=meta['args']['seed'],manifest_path=ROOT/meta['args']['manifest']); o,_=e.reset(a.seed); model=Model(features(o,c,True).shape[-1],len(critic_state(e))).to(dev); model.load_state_dict(ck['model']); model.eval(); rows=[]; start=time.perf_counter()
 for j in range(a.episodes):
  o,_=e.reset(a.seed+j); total=0.; n=0
  while not e.done:
   with torch.no_grad(): mu,_=model.policy(torch.as_tensor(features(o,c,True),device=dev)); act=20*mu.cpu().numpy()/np.sqrt(1+(mu.cpu().numpy()**2).sum(-1,keepdims=True)); best=o['prior_grid_m'][int(np.argmax(o['value_density']))]
   for i in range(c.n_uavs):
    d=best-o['positions_m'][i]; z=np.linalg.norm(d)
    if z>1e-6: act[i]=(1-a.guidance)*act[i]+a.guidance*d/z*20
   act=act*np.minimum(1,(20-1e-4)/np.maximum(np.linalg.norm(act,axis=1,keepdims=True),1e-12)); o,r,term,trunc,info=e.step(act); total+=r; n+=1
  rows.append({'episode':j,'track_id':str(e.episode['track_id']),'success_rate':info['success_rate'],'ttd_all':info['ttd_all'],'survival_score':info['survival_score'],'reward':total,'steps':n})
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'algorithm':'PEVA-v2 trained guidance','guidance':a.guidance,'checkpoint':a.checkpoint,'episodes':rows,'mean_success_rate':float(np.mean([x['success_rate'] for x in rows])),'mean_ttd_all':float(np.mean([x['ttd_all'] for x in rows])),'mean_survival_score':float(np.mean([x['survival_score'] for x in rows])),'manifest_sha256':meta['manifest_sha256'],'device':str(dev),'wall_s':time.perf_counter()-start},indent=2)); print(out)
if __name__=='__main__': main()
