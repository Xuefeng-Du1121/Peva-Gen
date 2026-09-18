import argparse,hashlib,json
from pathlib import Path
import numpy as np
from .real_dataset import RealDriftDataset
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=20260913); a=ap.parse_args(); p=Path(a.data); d=RealDriftDataset(p); s=d.split_by_track(seed=a.seed); ids={k:sorted(set(d.track_id[i] for i in v)) for k,v in s.items()}; payload={'data':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seed':a.seed,'rule':'grouped by drifter track_id','splits':{k:{'indices':v.tolist(),'track_ids':ids[k],'segments':len(v)} for k,v in s.items()},'warning':'tracks are observational drifters; not human survivor trajectories'}; o=Path(a.out); o.parent.mkdir(parents=True,exist_ok=True); o.write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
