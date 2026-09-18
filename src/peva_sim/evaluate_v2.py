import argparse,json,time
from pathlib import Path
import numpy as np,torch
from .env import Config
from .real_env import RealMaritimeSAR
from .mappo_v2 import Model,features,critic_state,ROOT
from .protocol import file_sha

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--episodes',type=int,default=20); ap.add_argument('--split',default='test'); ap.add_argument('--seed',type=int,default=90000); ap.add_argument('--out',required=True); args=ap.parse_args()
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ck=torch.load(args.checkpoint,map_location=device,weights_only=False); meta=ck['metadata']; c=Config(**{k:v for k,v in meta['config'].items() if k in Config.__dataclass_fields__}); e=RealMaritimeSAR(ROOT/meta['args']['data'],c,split=args.split,seed=meta['args']['seed'],manifest_path=ROOT/meta['args']['manifest']); o,_=e.reset(args.seed); model=Model(features(o,c,meta['args']['pvf']).shape[-1],len(critic_state(e))).to(device); model.load_state_dict(ck['model']); model.eval(); rows=[]; start=time.perf_counter()
 for j in range(args.episodes):
  o,_=e.reset(args.seed+j); total=0.; n=0
  while not e.done:
   with torch.no_grad(): mu,_=model.policy(torch.as_tensor(features(o,c,meta['args']['pvf']),device=device)); action=mu.cpu().numpy(); action=20*action/np.sqrt(1+(action*action).sum(-1,keepdims=True)); o,r,term,trunc,info=e.step(action); total+=r; n+=1
  rows.append({'episode':j,'track_id':str(e.episode['track_id']),'success_rate':info['success_rate'],'ttd_all':info['ttd_all'],'survival_score':info['survival_score'],'reward':total,'steps':n})
 out=(ROOT/args.out).resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'checkpoint':args.checkpoint,'split':args.split,'episodes':rows,'mean_success_rate':float(np.mean([x['success_rate'] for x in rows])),'mean_ttd_all':float(np.mean([x['ttd_all'] for x in rows])),'mean_survival_score':float(np.mean([x['survival_score'] for x in rows])),'device':str(device),'manifest_sha256':meta['manifest_sha256'],'wall_s':time.perf_counter()-start},indent=2)); print(out)
if __name__=='__main__': main()
