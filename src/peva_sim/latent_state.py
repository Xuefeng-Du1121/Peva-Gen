"""Learned target-state representation for the 128-D belief path.

Codec fitting uses training-only privileged target labels. Execution uses only
the decoder on sampled latent beliefs; encoding true targets is training-only.
The codec must be frozen before diffusion fitting to avoid a moving/collapsing
denoising target. This is an explicit implementation choice absent from the draft.
"""
import torch
from torch import nn

class TargetStateCodec(nn.Module):
    def __init__(self,n_targets=1,latent_dim=128):
        super().__init__()
        if n_targets<1 or latent_dim!=128:
            raise ValueError("Positive target count and paper latent dimension 128 required")
        self.n_targets=n_targets
        self.state_dim=2*n_targets
        self.latent_dim=latent_dim
        self.encoder=nn.Sequential(nn.Linear(self.state_dim,128),nn.SiLU(),
                                   nn.Linear(128,latent_dim),nn.LayerNorm(latent_dim,elementwise_affine=False))
        self.decoder=nn.Sequential(nn.Linear(latent_dim,128),nn.SiLU(),
                                   nn.Linear(128,self.state_dim))
    def encode_target(self,state):
        if state.shape[-1]!=self.state_dim or not torch.isfinite(state).all():
            raise ValueError("Invalid training target")
        return self.encoder(state)
    def decode(self,z):
        if z.shape[-1]!=self.latent_dim or not torch.isfinite(z).all():
            raise ValueError("Invalid belief latent")
        return self.decoder(z)
    def decode_locations_m(self,z,region_m):
        if region_m<=0: raise ValueError("Positive region scale required")
        return ((self.decode(z)+1)*(region_m/2)).reshape(*z.shape[:-1],self.n_targets,2)
    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self
    def pullback_location_gradient(self,z,gradient_x,region_m):
        """J_decoder^T grad_x, with metres included exactly once; no finite differences."""
        if gradient_x.shape!=(*z.shape[:-1],self.n_targets,2):
            raise ValueError("Location gradient shape mismatch")
        if not torch.isfinite(gradient_x).all(): raise ValueError("Nonfinite location gradient")
        with torch.enable_grad():
            leaf=z.detach().requires_grad_(True)
            xy=self.decode_locations_m(leaf,region_m)
            result=torch.autograd.grad(xy,leaf,grad_outputs=gradient_x.detach())[0]
        return result.detach()
