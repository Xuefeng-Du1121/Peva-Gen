import numpy as np
from .real_dataset import RealDriftDataset
class RealReplay:
 def __init__(self,path,split='test',seed=0,manifest_path=None):
  self.dataset=RealDriftDataset(path)
  if manifest_path is not None:
   from .protocol import manifest_indices
   self.indices=manifest_indices(self.dataset,path,manifest_path,split)
  else:
   self.indices=self.dataset.split_by_track(seed=seed)[split]
  self.cursor=0
 def reset(self,index=None):
  if index is None: index=self.cursor; self.cursor=(self.cursor+1)%len(self.indices)
  if not 0<=index<len(self.indices): raise IndexError(index)
  i=self.indices[index]; return {k:getattr(self.dataset,k)[i].copy() for k in ('track_id','times','positions_deg','velocities_ms','environment_uv_ms')}
 def summary(self):
  x=self.dataset.subset(self.indices); return {'episodes':len(self.indices),'unique_tracks':len(set(x['track_id'])),'position_deg_min':x['positions_deg'].min((0,1)).tolist(),'position_deg_max':x['positions_deg'].max((0,1)).tolist(),'environment_uv_mean':x['environment_uv_ms'].mean((0,1)).tolist(),'step_seconds':3600}
