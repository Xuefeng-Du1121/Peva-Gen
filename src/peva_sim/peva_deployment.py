import torch
from .joint_modules import JointBeliefCommunication
class PEVABeliefDeployment:
 """Inference-only adapter; context is actor-visible, target labels excluded."""
 def __init__(self,checkpoint,device='cpu'):
  self.device=torch.device(device); self.model=JointBeliefCommunication().to(self.device); ck=torch.load(checkpoint,map_location=self.device,weights_only=False); self.model.load_state_dict(ck['model']); self.model.eval(); self.source=ck.get('data_sha256')
 @torch.no_grad()
 def infer(self,context,samples=2):
  context=torch.as_tensor(context,dtype=torch.float32,device=self.device)
  if context.ndim!=2 or context.shape[1]!=16: raise ValueError('context must be (batch,16)')
  out=self.model.deployment(context); belief=self.model.belief.sample(context,samples=samples)
  return {'message':out['message'],'belief_samples':belief,'uncertainty':belief.var(1).mean(-1),'kl_rate':out['kl']}
