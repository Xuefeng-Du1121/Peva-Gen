from pathlib import Path
import numpy as np
class RealDriftDataset:
 def __init__(self,path):
  d=np.load(Path(path),allow_pickle=False); self.track_id=d['track_id'].astype(str); self.times=d['times']; self.positions_deg=d['positions_deg']; self.velocities_ms=d['velocities_ms']; self.environment_uv_ms=d['environment_uv_ms']; n=len(self.track_id)
  if self.times.shape!=(n,4) or self.positions_deg.shape!=(n,4,2): raise ValueError('invalid collocation shapes')
 def split_by_track(self,train=.6,val=.2,seed=0):
  if not 0<train<1 or not 0<=val<1 or train+val>=1: raise ValueError('invalid split')
  ids=np.unique(self.track_id); np.random.default_rng(seed).shuffle(ids); a=int(len(ids)*train); b=int(len(ids)*(train+val)); return {k:np.flatnonzero(np.isin(self.track_id,v)) for k,v in {'train':ids[:a],'validation':ids[a:b],'test':ids[b:]}.items()}
 def subset(self,idx): return {k:getattr(self,k)[idx] for k in ('track_id','times','positions_deg','velocities_ms','environment_uv_ms')}
