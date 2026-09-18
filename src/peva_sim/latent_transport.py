"""128-D transport belief; separate architecture, never load 2-D weights here."""
import torch
from torch import nn
from .joint_transport import aggregate_messages,quantize_message
from .peva_gen import ValueAlignedBottleneck
from .residual_diffusion import ResidualDiffusionBelief
from .latent_state import TargetStateCodec

class LatentTransportJointBelief(nn.Module):
    architecture="transport-latent128-frozen-target-codec-v1"
    def __init__(self,context_dim,codec):
        super().__init__()
        if not isinstance(codec,TargetStateCodec): raise ValueError("Trained target codec required")
        self.context_dim=context_dim
        self.codec=codec.freeze()
        self.encoder=ValueAlignedBottleneck(context_dim,32)
        self.decoder=nn.Sequential(nn.Linear(32,128),nn.SiLU(),nn.Linear(128,128))
        self.belief=ResidualDiffusionBelief(128,context_dim+32,steps=5)
    def train(self,mode=True):
        super().train(mode)
        self.codec.eval()
        return self
    def loss(self,context,target,score,received,rate_correction):
        b,n,d=context.shape
        if d!=self.context_dim or target.shape!=(b,self.codec.state_dim):
            raise ValueError("Context/target contract mismatch")
        if rate_correction.shape!=(b,) or not torch.isfinite(rate_correction).all() or (rate_correction<=0).any():
            raise ValueError("Finite positive exact replay correction required")
        with torch.no_grad():
            z=self.codec.encode_target(target)
        msg,kl=self.encoder(context.reshape(b*n,d))
        aggregate=aggregate_messages(quantize_message(msg).reshape(b,n,32),score.detach(),received)
        targets=z[:,None,:].expand(b,n,128)
        relevance=(self.decoder(aggregate)-targets).square().mean()
        condition=torch.cat((context,aggregate),-1).reshape(b*n,-1)
        denoising=self.belief.training_loss(targets.reshape(b*n,128),condition)
        rate=(rate_correction.detach()*kl.reshape(b,n).sum(-1)).mean()
        return dict(relevance=relevance,belief=denoising,rate=rate,aggregate=aggregate)
    @staticmethod
    def objective(losses,rate_weight=1e-3):
        return losses["relevance"]+losses["belief"]+rate_weight*losses["rate"]
    @torch.no_grad()
    def sample(self,local_context,delivered_aggregate,samples=2,initial_noise=None):
        if local_context.ndim!=2 or local_context.shape[1]!=self.context_dim:
            raise ValueError("Local context shape mismatch")
        if delivered_aggregate.shape!=(len(local_context),32):
            raise ValueError("Delivered aggregate shape mismatch")
        condition=torch.cat((local_context,delivered_aggregate),-1)
        z=self.belief.sample(condition,samples=samples,initial_noise=initial_noise)
        return z,self.codec.decode(z)
