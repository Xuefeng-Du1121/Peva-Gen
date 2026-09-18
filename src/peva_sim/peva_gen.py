"""PEVA-Gen stage-1 modules.
The module is intentionally isolated from MAPPO until the baseline evaluation is frozen.
"""
import torch
from torch import nn

class ReliabilityFusion(nn.Module):
    """Physics/critic value weight with explicit deployment modes."""
    def __init__(self,beta_min=.1,mode='fixed'):
        super().__init__(); self.beta_min=float(beta_min); self.mode=mode
        if mode not in ('fixed','adaptive'): raise ValueError(mode)
        self.confidence=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,critic_grad,physics_grad,align=None,ensemble=None,td_error=None):
        if critic_grad.shape != physics_grad.shape: raise ValueError('gradient shape mismatch')
        if self.mode=='fixed': beta=torch.full_like(critic_grad[...,:1],self.beta_min)
        else:
            if align is None or ensemble is None or td_error is None: raise ValueError('adaptive confidence inputs required')
            q=torch.sigmoid(4*align+2*ensemble-4*td_error).unsqueeze(-1); beta=self.beta_min+(1-self.beta_min)*(1-q)
        weight=(1-beta)*critic_grad.norm(dim=-1,keepdim=True)+beta*physics_grad.norm(dim=-1,keepdim=True)
        return weight.clamp(.1,10),beta

class ValueAlignedBottleneck(nn.Module):
    """Variational message encoder; KL is reported separately from actual bytes."""
    def __init__(self,obs_dim,message_dim=32):
        super().__init__(); self.mu=nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),nn.Linear(128,message_dim)); self.logvar=nn.Sequential(nn.Linear(obs_dim,128),nn.ReLU(),nn.Linear(128,message_dim))
    def forward(self,obs):
        mu=self.mu(obs); logvar=self.logvar(obs).clamp(-8,8); std=(.5*logvar).exp(); z=mu+std*torch.randn_like(std) if self.training else mu
        kl=.5*(mu.pow(2)+logvar.exp()-1-logvar).sum(-1); return z,kl
    @staticmethod
    def quantized_bytes(z,bits=8,header_bytes=16):
        if bits not in (8,16): raise ValueError(bits)
        return z.shape[-1]*(bits//8)+header_bytes
