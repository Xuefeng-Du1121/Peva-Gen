"""Public-grid multi-target value pullback; no world-state access.

Caller must supply confirmed target IDs received through the shared protocol.
This helper does not infer confirmations from privileged simulator state.
It does not resolve the paper's deployment critic/fusion choice.
"""
import numpy as np
import torch
from .message_relevance import public_grid_gradient

def public_multitarget_gradient(codec,z,observation,confirmed,config):
    if z.ndim!=2 or z.shape[1]!=128 or codec.n_targets!=config.n_targets:
        raise ValueError("Expected batch of matching multi-target latents")
    mask=np.asarray(confirmed)
    if mask.dtype!=np.bool_ or mask.shape!=(config.n_targets,):
        raise ValueError("Explicit confirmed target mask required")
    with torch.no_grad():
        xy=codec.decode_locations_m(z,config.region_m).cpu().numpy()
    gx=public_grid_gradient(observation,xy.reshape(-1,2)).reshape(xy.shape)
    gx[:,mask,:]=0.
    reference=1/(2*np.pi*config.prior_sigma_m**2)
    gradient=torch.as_tensor(gx/reference,dtype=z.dtype,device=z.device)
    return codec.pullback_location_gradient(z,gradient,config.region_m)
