"""Stable residual parameterization of the paper's epsilon predictor.

eps_theta(z_k,k,c) = sqrt(alpha_bar)*v_theta(z_k,k,c)
                    + sqrt(1-alpha_bar)*z_k
This is an invertible parameterization for 0<alpha_bar<1. The training objective
remains epsilon MSE (NOT unweighted v MSE). Reconstruct x0 algebraically without
dividing by tiny sqrt(alpha_bar). Separate architecture: legacy checkpoints
must never be interpreted as residual-network weights.
"""
import torch
from .diffusion_belief import DiffusionBelief


class ResidualDiffusionBelief(DiffusionBelief):
    def training_loss(self,z0,context,k=None,noise=None,reduction="mean"):
        if context.shape!=(len(z0),self.context_dim):
            raise ValueError("context shape mismatch")
        if k is None: k=torch.randint(self.training_steps,(len(z0),),device=z0.device)
        if noise is None: noise=torch.randn_like(z0)
        zk=self.q_sample(z0,k,noise)
        ab=self.alpha_bar[k,None]
        v_target=ab.sqrt()*noise-(1-ab).sqrt()*z0
        # Algebraically identical epsilon error, evaluated without cancellation.
        losses=(ab*(self.eps(zk,k,context)-v_target).square()).mean(-1)
        if reduction=="none": return losses
        if reduction=="mean": return losses.mean()
        raise ValueError("invalid reduction")

    @torch.no_grad()
    def sample(self,context,samples=2,guidance=None,initial_noise=None):
        if guidance is not None:
            raise ValueError("residual inference guidance not yet validated; disabled")
        if context.ndim!=2 or context.shape[1]!=self.context_dim or samples<1:
            raise ValueError("invalid context/sample count")
        batch=len(context); context=context.repeat_interleave(samples,0)
        if initial_noise is None:
            z=torch.randn(batch*samples,self.z_dim,device=context.device,dtype=context.dtype)
        else:
            if initial_noise.shape!=(batch,samples,self.z_dim):
                raise ValueError("initial noise shape mismatch")
            z=initial_noise.reshape(batch*samples,self.z_dim).clone()
        schedule=torch.linspace(self.training_steps-1,0,self.steps,device=z.device).round().long()
        for j,k in enumerate(schedule):
            ab=self.alpha_bar[k]
            v=self.eps(z,k.expand(len(z)),context)
            predicted_noise=ab.sqrt()*v+(1-ab).sqrt()*z
            x0=ab.sqrt()*z-(1-ab).sqrt()*v
            previous_ab=self.alpha_bar[schedule[j+1]] if j+1<len(schedule) else z.new_tensor(1.)
            z=previous_ab.sqrt()*x0+(1-previous_ab).sqrt()*predicted_noise
        return z.reshape(batch,samples,self.z_dim)
