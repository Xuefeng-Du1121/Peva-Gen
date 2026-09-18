"""Analytic particle-KDE PVF value/gradient, matching env.pvf (single target)."""
import numpy as np


def value_gradient(env, xy):
    x=np.asarray(xy,dtype=float).reshape(-1,2)
    c=env.cfg
    h=max(c.sensing_m,c.prior_sigma_m*c.particles**(-1/6))
    value=np.zeros(len(x)); gradient=np.zeros_like(x)
    factor=c.detection_p*np.exp(-env.t/c.survival_tau_s)/(2*np.pi*h*h)
    for g in range(c.n_targets):
        if env.found[g]: continue
        delta=x[:,None,:]-env.particles[g][None,:,:]
        mass=np.exp(-.5*(delta**2).sum(-1)/h**2)*env.weights[g][None,:]
        value+=factor*mass.sum(-1)
        gradient+=factor*(-delta/h**2*mass[:,:,None]).sum(1)
    return value,gradient


PRIORITY_SCHEMA = "fixed-density-unit-pvf-gradient-v2"


def latent_gradient(env, target_m):
    """Fixed density unit preserves survival and coverage amplitude.

    Reference unit is 1/(2*pi*initial_prior_sigma^2), never current field peak.
    Chain rule uses z=2*x/region-1. This numeric convention is explicit.
    """
    _,grad=value_gradient(env,target_m)
    scale=1/(2*np.pi*env.cfg.prior_sigma_m**2)
    return grad*(env.cfg.region_m/2)/scale


def normalized_priority(env, target_m, observation):
    norm=np.linalg.norm(latent_gradient(env,target_m),axis=-1)
    return float(np.clip(norm.mean(),.1,10))
