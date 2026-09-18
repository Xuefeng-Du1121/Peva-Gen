"""128-D neural-decoder coupled update (paper Sections 4.2-4.5).

All sample corrections are attached to the actual frozen draw probabilities.
Actor inputs/raw actions/log-probabilities are cached rollout quantities; changing
auxiliary parameters never retroactively changes the behavior policy features.
This is block-coordinate coupling, not an end-to-end actor-through-belief gradient.
"""
import numpy as np
import torch
from .value_replay import ValueReplaySampler
from .ppo_math import latent_log_prob


def critic_latent_gradient(critic,state,n_uavs,codec,create_graph=False):
    """Gradient at decoded target; critic fit remains at recorded true state."""
    if state.ndim!=2 or state.shape[1]!=2*n_uavs+3*codec.n_targets+1:
        raise ValueError("Single-target critic state contract mismatch")
    end=2*n_uavs+codec.state_dim
    target=state[:,2*n_uavs:end]*2-1
    z=codec.encode_target(target).detach().requires_grad_(True)
    normalized=(codec.decode(z)+1)/2
    decoded_state=torch.cat((state[:,:2*n_uavs].detach(),normalized,state[:,end:].detach()),-1)
    surrogate=critic(decoded_state).squeeze(-1)
    gradient=torch.autograd.grad(surrogate.sum(),z,create_graph=create_graph)[0]
    return critic(state.detach()).squeeze(-1),gradient


def stable_cosine(a,b,epsilon=1e-6):
    return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(epsilon)


def fused_priority(critic_gradient,physics_gradient,beta,alpha=1.):
    if critic_gradient.shape!=physics_gradient.shape or not 0<=beta<=1 or alpha<=0:
        raise ValueError("invalid fusion inputs")
    return ((1-beta)*critic_gradient.norm(dim=-1)+beta*physics_gradient.norm(dim=-1)).clamp(.1,10).pow(alpha)


class LatentCoupledUpdater:
    def __init__(self,policy,auxiliary,policy_optimizer,auxiliary_optimizer,
                 n_uavs,seed=0,beta_min=.1,risk_lambda=1.,alignment_weight=.1,fixed_beta=None):
        self.policy=policy;self.auxiliary=auxiliary
        if not np.isfinite([beta_min,risk_lambda,alignment_weight]).all() or not 0<=beta_min<=1 or risk_lambda<0 or alignment_weight<0:
            raise ValueError("invalid confidence/risk/alignment configuration")
        self.policy_optimizer=policy_optimizer;self.auxiliary_optimizer=auxiliary_optimizer
        self.n_uavs=n_uavs;self.beta_min=beta_min;self.risk_lambda=risk_lambda
        self.alignment_weight=alignment_weight;self.rng=np.random.default_rng(seed)
        if fixed_beta is not None and (not np.isfinite(fixed_beta) or not 0<=fixed_beta<=1):
            raise ValueError("fixed_beta must be absent or within [0,1]")
        self.fixed_beta=fixed_beta
        self.ema_td=None

    def update(self,rollout,batch_size):
        required=("state","features","raw","old_log_prob","advantage","returns",
                  "td_error","physics_gradient","context","target","score","received","spread")
        if any(k not in rollout for k in required):
            raise ValueError("incomplete coupled rollout")
        r=rollout;size=len(r["state"])
        if size<1 or batch_size<1: raise ValueError("empty update")
        for k in required:
            if len(r[k])!=size: raise ValueError("rollout length mismatch: "+k)
            if not torch.isfinite(r[k]).all(): raise ValueError("nonfinite rollout: "+k)
        # Freeze sampling priorities over the whole on-policy collection.
        _,cg=critic_latent_gradient(self.policy.critic,r["state"],self.n_uavs,self.auxiliary.codec)
        if (r["spread"]<0).any():
            raise ValueError("posterior spread cannot be negative")
        pg=r["physics_gradient"].detach()
        with torch.no_grad():
            mu,ls=self.policy.policy(r["features"])
            ratio_error=float((torch.exp(latent_log_prob(mu,ls,r["raw"])-r["old_log_prob"])-1).abs().max())
        if ratio_error>1e-5:
            raise ValueError("stale behavior policy: collect fresh data before coupled update")
        alignment=float(stable_cosine(cg.detach(),pg).mean())
        observed=float(r["td_error"].detach().abs().mean())
        self.ema_td=observed if self.ema_td is None else .95*self.ema_td+.05*observed
        normalized_td=self.ema_td/(1+self.ema_td)  # fixed unit reward scale
        confidence=float(torch.sigmoid(torch.tensor(4*alignment-4*normalized_td)))
        beta=self.beta_min+(1-self.beta_min)*(1-confidence)
        if self.fixed_beta is not None: beta=self.fixed_beta
        priority=fused_priority(cg.detach(),pg,beta)
        sampler=ValueReplaySampler(priority.cpu().numpy())
        index,correction=sampler.sample(self.rng,batch_size)
        idx=torch.as_tensor(index,device=r["state"].device)
        corr=torch.as_tensor(correction,dtype=r["state"].dtype,device=idx.device)
        # Same sampled transition indices for all three stages.
        self.auxiliary.train()
        aux=self.auxiliary.loss(r["context"][idx],r["target"][idx],r["score"][idx],
                               r["received"][idx],corr)
        auxiliary_loss=self.auxiliary.objective(aux)
        mu,ls=self.policy.policy(r["features"][idx].detach())
        ratio=torch.exp(latent_log_prob(mu,ls,r["raw"][idx])-r["old_log_prob"][idx])
        advantage=r["advantage"][idx,None].detach()
        spread=r["spread"][idx].detach().mean(-1)
        # Same formula without cancellation when tanh rounds to one.
        gate=2*torch.sigmoid(-2*self.risk_lambda*spread)
        risk_weight=corr.clamp(.2,5)*gate
        surrogate=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean(-1)
        actor_loss=-(risk_weight*surrogate).mean()
        value,critic_grad=critic_latent_gradient(self.policy.critic,r["state"][idx],self.n_uavs,self.auxiliary.codec,create_graph=True)
        critic_fit=.5*(corr*(value-r["returns"][idx].detach()).square()).mean()
        direction_loss=(1-stable_cosine(critic_grad,pg[idx])).mean()
        policy_loss=actor_loss+critic_fit+self.alignment_weight*beta*direction_loss
        if not torch.isfinite(auxiliary_loss+policy_loss):
            raise FloatingPointError("nonfinite coupled objective")
        self.auxiliary_optimizer.zero_grad();self.policy_optimizer.zero_grad()
        auxiliary_loss.backward();policy_loss.backward()
        aux_norm=torch.nn.utils.clip_grad_norm_(self.auxiliary.parameters(),1.)
        policy_norm=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),.5)
        if not torch.isfinite(aux_norm+policy_norm):
            raise FloatingPointError("nonfinite coupled gradient")
        self.auxiliary_optimizer.step();self.policy_optimizer.step()
        return {"beta":beta,"confidence":confidence,"ema_td":self.ema_td,
                "priority_min":float(priority.min()),"priority_max":float(priority.max()),
                "correction_mean":float(corr.mean()),"risk_gate_mean":float(gate.mean()),
                "spread_mean":float(spread.mean()),"spread_max":float(spread.max()),
                "risk_gate_zero_fraction":float((gate==0).float().mean()),
                "behavior_ratio_error":ratio_error,"actor_loss":float(actor_loss.detach()),
                "critic_fit":float(critic_fit.detach()),"direction_loss":float(direction_loss.detach()),
                **{"aux_"+k:float(aux[k].detach()) for k in ("belief","relevance","rate")}}
