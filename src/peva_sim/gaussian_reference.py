"""Training-distribution Gaussian reference plus an unrestricted learned v residual.

A numerical parameterization/initialization, not posterior clipping. For any
reference r(z,k), v=r+delta_v retains the same epsilon prediction objective.
Gaussian moments must be fitted from training targets only and frozen.
"""
import torch
from torch import nn
from .diffusion_belief import EpsilonNet

class GaussianReferenceVNet(nn.Module):
    def __init__(self,z_dim,context_dim,mean=None,covariance=None,training_steps=1000):
        super().__init__()
        mean=torch.zeros(z_dim,dtype=torch.float64) if mean is None else mean.detach().double()
        covariance=torch.eye(z_dim,dtype=torch.float64) if covariance is None else covariance.detach().double()
        if mean.shape!=(z_dim,) or covariance.shape!=(z_dim,z_dim):
            raise ValueError("Gaussian moment shape mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
            raise ValueError("Nonfinite Gaussian moments")
        eig,basis=torch.linalg.eigh((covariance+covariance.T)/2)
        if eig.min() < -1e-8*max(1.,float(eig.abs().max())):
            raise ValueError("Covariance must be positive semidefinite")
        self.register_buffer("mean",mean.float())
        self.register_buffer("eigenvalues",eig.clamp_min(0).float())
        self.register_buffer("basis",basis.float())
        self.residual=EpsilonNet(z_dim,context_dim,training_steps=training_steps)
        nn.init.zeros_(self.residual.net[-1].weight)
        nn.init.zeros_(self.residual.net[-1].bias)
        self.register_buffer("alpha_bar",torch.empty(training_steps))
    def reference(self,z,k):
        a=self.alpha_bar[k,None]
        b=1-a
        denominator=a*self.eigenvalues+b
        delta=z-a.sqrt()*self.mean
        coefficient=(a*b).sqrt()*(1-self.eigenvalues)/denominator
        return ((delta@self.basis)*coefficient)@self.basis.T-b.sqrt()*self.mean
    def forward(self,z,k,context):
        return self.reference(z,k)+self.residual(z,k,context)
