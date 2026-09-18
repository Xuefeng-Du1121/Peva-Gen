"""Numerical contracts for on-policy training."""
import numpy as np
import torch

def gae(rewards, values, next_values, bootstrap, continuation, gamma=.99, lam=.95):
    # bootstrap=0 for finite mission end; continuation=0 at every reset.
    # Collector cuts alone do not change next-state bootstrapping.
    r,v,n,b,c=[np.asarray(x,dtype=np.float64) for x in
                (rewards,values,next_values,bootstrap,continuation)]
    if not (r.shape==v.shape==n.shape==b.shape==c.shape):
        raise ValueError("GAE shape mismatch")
    out=np.zeros_like(r); acc=np.zeros_like(r[0])
    for i in reversed(range(len(r))):
        delta=r[i]+gamma*b[i]*n[i]-v[i]
        acc=delta+gamma*lam*c[i]*acc
        out[i]=acc
    return out.astype(np.float32),(out+v).astype(np.float32)

def disk_action(raw, speed):
    # Bijective R^2 -> open speed disk. No many-to-one radial clipping.
    return speed*raw/torch.sqrt(1+(raw*raw).sum(-1,keepdim=True))

def latent_log_prob(mu,logstd,raw):
    # Ratios use stored latent samples. Action transform is fixed, so Jacobians cancel.
    return torch.distributions.Normal(mu,logstd.exp()).log_prob(raw).sum(-1)
