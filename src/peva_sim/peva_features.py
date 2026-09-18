import numpy as np,torch
from .peva_deployment import PEVABeliefDeployment
from .mappo_v2 import features
class PEVAFeatureExtractor:
 def __init__(self,checkpoint,c,pvf=True):
  raise ValueError(
   'Legacy PEVA auxiliary features are invalid: training/deployment context '
   'semantics differ and messages bypass the radio. Retrain with a shared '
   'versioned context and transport-aware adapter; see '
   'docs/auxiliary-contract-audit-20260913.md. Baseline runs without --peva-aux remain available.')
 def __call__(self,obs):
  base=features(obs,self.c,self.pvf); contexts=[]
  for i in range(self.c.n_uavs):
   q=np.zeros(16,np.float32); q[:2]=obs['positions_m'][i]/self.c.region_m; q[2]=obs['time_s']/self.c.horizon_s; q[3]=float(obs['value_density'].max()); contexts.append(q)
  out=self.model.infer(np.asarray(contexts),samples=2); msg=out['message'].numpy(); belief=out['belief_samples'].mean(1).numpy(); unc=out['uncertainty'].numpy()[:,None]
  return np.concatenate((base,msg,belief,unc),axis=1).astype(np.float32)
