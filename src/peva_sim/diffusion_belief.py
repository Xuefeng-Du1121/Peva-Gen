"""Conditional epsilon diffusion and deterministic DDIM for belief reconstruction.

steps is the inference budget, not the number of training noise levels.
Unweighted per-sample loss supports value-prioritized replay without double weighting.
"""
import math
import torch
from torch import nn

class EpsilonNet(nn.Module):
    def __init__(self,z_dim=2,context_dim=16,hidden=128,training_steps=1000):
        super().__init__()
        self.training_steps=training_steps
        self.register_buffer("frequencies",torch.exp(torch.linspace(0,math.log(1000),8)))
        self.net=nn.Sequential(nn.Linear(z_dim+context_dim+16,hidden),nn.SiLU(),
                               nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,z_dim))
    def forward(self,z,k,context):
        phase=(k.float()/(self.training_steps-1))[:,None]*self.frequencies[None]
        embedding=torch.cat((phase.sin(),phase.cos()),-1)
        return self.net(torch.cat((z,context,embedding),-1))

class DiffusionBelief(nn.Module):
    def __init__(self,z_dim=2,context_dim=16,steps=5,training_steps=1000):
        super().__init__()
        if not 1<=steps<=training_steps or training_steps<2:
            raise ValueError("Invalid sampling/training steps")
        self.steps=steps; self.training_steps=training_steps
        self.z_dim=z_dim; self.context_dim=context_dim
        grid=torch.linspace(0,1,training_steps+1,dtype=torch.float64)
        cumulative=torch.cos((grid+.008)/1.008*math.pi/2).square()
        cumulative=cumulative/cumulative[0]
        beta=(1-cumulative[1:]/cumulative[:-1]).clamp(1e-8,.999)
        self.register_buffer("beta",beta.float())
        self.register_buffer("alpha_bar",torch.cumprod(1-beta,0).float())
        self.eps=EpsilonNet(z_dim,context_dim,training_steps=training_steps)

    def q_sample(self,z0,k,noise):
        if z0.ndim!=2 or z0.shape[-1]!=self.z_dim or noise.shape!=z0.shape:
            raise ValueError("Expected (batch,z_dim) clean states and matching noise")
        if k.shape!=(len(z0),) or k.dtype!=torch.long:
            raise ValueError("Expected one integer noise timestep per sample")
        if torch.any(k<0) or torch.any(k>=self.training_steps):
            raise ValueError("Noise timestep out of range")
        ab=self.alpha_bar[k,None]
        return ab.sqrt()*z0+(1-ab).sqrt()*noise

    def training_loss(self,z0,context,k=None,noise=None,reduction="mean"):
        if context.shape!=(len(z0),self.context_dim):
            raise ValueError("Context shape mismatch")
        if k is None:
            k=torch.randint(self.training_steps,(len(z0),),device=z0.device)
        if noise is None: noise=torch.randn_like(z0)
        zk=self.q_sample(z0,k,noise)
        losses=(self.eps(zk,k,context)-noise).square().mean(-1)
        if reduction=="none": return losses
        if reduction=="mean": return losses.mean()
        raise ValueError("reduction must be mean or none")

    @torch.no_grad()
    def sample(self,context,samples=2,guidance=None,initial_noise=None):
        if context.ndim!=2 or context.shape[1]!=self.context_dim or samples<1:
            raise ValueError("Invalid context or sample count")
        batch=len(context); context=context.repeat_interleave(samples,0)
        if initial_noise is None:
            z=torch.randn(batch*samples,self.z_dim,device=context.device,dtype=context.dtype)
        else:
            if initial_noise.shape!=(batch,samples,self.z_dim):
                raise ValueError("initial_noise shape mismatch")
            z=initial_noise.reshape(batch*samples,self.z_dim).clone()
        schedule=torch.linspace(self.training_steps-1,0,self.steps,device=z.device).round().long()
        for j,k in enumerate(schedule):
            t=k.expand(len(z)); ab=self.alpha_bar[k]
            predicted_noise=self.eps(z,t,context)
            if guidance is not None:
                # Caller must supply a noisy-state log-density gradient.
                # An arbitrary clean PVF gradient is not an exact posterior tilt.
                with torch.enable_grad():
                    gradient=guidance(z.detach().requires_grad_(True)).detach()
                if gradient.shape!=z.shape: raise ValueError("Guidance shape mismatch")
                predicted_noise=predicted_noise-(1-ab).sqrt()*gradient
            x0=(z-(1-ab).sqrt()*predicted_noise)/ab.sqrt()
            previous_ab=self.alpha_bar[schedule[j+1]] if j+1<len(schedule) else z.new_tensor(1.)
            z=previous_ab.sqrt()*x0+(1-previous_ab).sqrt()*predicted_noise
        return z.reshape(batch,samples,self.z_dim)
