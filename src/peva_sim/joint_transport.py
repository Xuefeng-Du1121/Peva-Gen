"""Transport-aware differentiable CIB and belief training, paper Sections 4.3-4.4.

This module consumes explicit received-link masks. It is not a complete policy
trainer. Replay must sample by priority ONCE; pass its exact inverse probability
correction for rate. Never multiply relevance or denoising loss by priority again.
"""
import torch
from torch import nn
from .diffusion_belief import DiffusionBelief
from .peva_gen import ValueAlignedBottleneck


def quantize_message(message):
    """42-byte codec's dequantized forward value with straight-through gradient.

    The backward estimator is an approximation; KL is a variational information
    penalty, not a measurement of fixed-length transmitted bytes.
    """
    if message.shape[-1] != 32 or not torch.isfinite(message).all():
        raise ValueError("codec requires 32 finite components")
    x = message.detach().double()
    scale = (x.abs().amax(-1,keepdim=True)/127).clamp_min(torch.finfo(torch.float32).tiny).float()
    if not torch.isfinite(scale).all():
        raise ValueError("codec scale overflow")
    code = torch.round(x/scale.double()).clamp(-127,127)
    reconstructed = (code*scale.double()).to(message.dtype)
    return message + (reconstructed-message).detach()


def aggregate_messages(message, score, received):
    """Inputs B,N,D; B,N log-weights; B,N(receiver),N(sender) delivery mask.

    Self information is always retained (paper N(i) union {i}). Scores must
    originate from observable estimates at execution, not privileged targets.
    """
    if message.ndim != 3:
        raise ValueError("messages must have batch,agent,feature axes")
    b,n,_ = message.shape
    if score.shape != (b,n) or received.shape != (b,n,n) or received.dtype != torch.bool:
        raise ValueError("score/mask shape or type mismatch")
    if not torch.isfinite(message).all() or not torch.isfinite(score).all():
        raise ValueError("nonfinite message/score")
    mask = received | torch.eye(n, dtype=torch.bool, device=message.device)[None]
    logits = score[:,None,:].expand(b,n,n).masked_fill(~mask, -torch.inf)
    weights = torch.softmax(logits, -1)
    return torch.bmm(weights,message)


class TransportJointBelief(nn.Module):
    def __init__(self, context_dim, message_dim=32, diffusion_parameterization="epsilon"):
        super().__init__()
        self.context_dim = context_dim
        self.encoder = ValueAlignedBottleneck(context_dim, message_dim)
        self.decoder = nn.Sequential(nn.Linear(message_dim,128),nn.SiLU(),nn.Linear(128,2))
        if diffusion_parameterization=="epsilon":
            self.belief = DiffusionBelief(2,context_dim+message_dim,steps=5)
        elif diffusion_parameterization=="v_residual":
            from .residual_diffusion import ResidualDiffusionBelief
            self.belief = ResidualDiffusionBelief(2,context_dim+message_dim,steps=5)
        else:
            raise ValueError("unknown diffusion parameterization")

    def loss(self, context, target, score, received, rate_correction):
        b,n,d = context.shape
        if d != self.context_dim or target.shape != (b,2):
            raise ValueError("context/target contract mismatch")
        if rate_correction.shape != (b,) or not torch.isfinite(rate_correction).all() or (rate_correction<=0).any():
            raise ValueError("need finite positive exact replay correction per transition")
        msg, kl = self.encoder(context.reshape(b*n,d))
        aggregate = aggregate_messages(quantize_message(msg).reshape(b,n,-1),score.detach(),received)
        target_agents = target[:,None,:].expand(b,n,2)
        relevance = (self.decoder(aggregate)-target_agents).square().mean()
        condition = torch.cat((context,aggregate),-1).reshape(b*n,-1)
        denoising = self.belief.training_loss(target_agents.reshape(b*n,2),condition)
        rate = (rate_correction.detach()*kl.reshape(b,n).sum(-1)).mean()
        return {"relevance":relevance, "belief":denoising, "rate":rate,
                "aggregate":aggregate}

    @staticmethod
    def objective(losses, rate_weight=1e-3):
        return losses["relevance"]+losses["belief"]+rate_weight*losses["rate"]
