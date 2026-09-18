import torch
from torch import nn
from .diffusion_belief import DiffusionBelief
from .peva_gen import ValueAlignedBottleneck
class CIBDecoder(nn.Module):
 def __init__(self,message_dim=32,state_dim=2): super().__init__(); self.net=nn.Sequential(nn.Linear(message_dim,64),nn.ReLU(),nn.Linear(64,state_dim))
 def forward(self,z): return self.net(z)
class JointBeliefCommunication(nn.Module):
 def __init__(self,context_dim=16,z_dim=2,message_dim=32,rate_weight=1e-3):
  super().__init__(); self.belief=DiffusionBelief(z_dim,context_dim,steps=5); self.encoder=ValueAlignedBottleneck(context_dim,message_dim); self.decoder=CIBDecoder(message_dim,z_dim); self.rate_weight=rate_weight
 def loss(self,target,context,value_weight=None):
  denoise=self.belief.training_loss(target,context,reduction='none'); msg,kl=self.encoder(context); recon=(self.decoder(msg)-target).square().mean(-1)
  if value_weight is None: value_weight=torch.ones_like(recon)
  if value_weight.shape!=recon.shape: raise ValueError('value weight shape mismatch')
  relevance=(value_weight*recon).mean(); rate=self.rate_weight*kl.mean(); return {'total':relevance+rate+denoise.mean(),'belief':denoise.mean(),'relevance':relevance,'rate':rate,'kl':kl.mean(),'message':msg,'reconstruction':self.decoder(msg)}
 def deployment(self,context):
  self.eval()
  with torch.no_grad(): msg,kl=self.encoder(context); return {'message':msg,'kl':kl,'reconstruction':self.decoder(msg)}
