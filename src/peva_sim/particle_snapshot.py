"""Reconstruct training PVF gradients at neural-decoded multi-target locations."""
from types import SimpleNamespace
import numpy as np
import torch
from .physics_priority import value_gradient

def snapshot_environment(config,particles,weights,found,time_s):
    p=np.asarray(particles);w=np.asarray(weights);f=np.asarray(found)
    if p.shape!=(config.n_targets,config.particles,2) or w.shape!=p.shape[:-1] or f.shape!=(config.n_targets,):
        raise ValueError("Particle snapshot shape mismatch")
    if f.dtype!=np.bool_ or not np.isfinite(p).all() or not np.isfinite(w).all() or (w<0).any():
        raise ValueError("Invalid particle snapshot")
    if not np.isfinite(time_s) or not 0<=time_s<=config.horizon_s: raise ValueError("Invalid snapshot time")
    return SimpleNamespace(cfg=config,particles=p,weights=w,found=f,t=float(time_s))

def decoded_physics_gradient(codec,z,snapshot):
    """Sum of shared PVF at active decoded slots, pulled back to one latent."""
    if z.shape!=(1,128) or codec.n_targets!=snapshot.cfg.n_targets:
        raise ValueError("One matching multi-target latent required")
    with torch.no_grad():
        locations=codec.decode_locations_m(z,snapshot.cfg.region_m).cpu().numpy()[0]
    _,gx=value_gradient(snapshot,locations)
    gx[snapshot.found]=0.
    reference=1/(2*np.pi*snapshot.cfg.prior_sigma_m**2)
    gradient=torch.as_tensor(gx[None]/reference,dtype=z.dtype,device=z.device)
    return codec.pullback_location_gradient(z,gradient,snapshot.cfg.region_m)
