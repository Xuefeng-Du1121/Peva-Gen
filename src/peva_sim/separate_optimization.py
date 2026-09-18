"""Separate actor/critic optimization for disjoint feed-forward MAPPO networks."""
import torch

def step_separate(actor_loss,critic_loss,actor_parameters,critic_parameters,actor_optimizer,critic_optimizer,max_norm=.5):
    actor=list(actor_parameters);critic=list(critic_parameters)
    if {id(p) for p in actor}&{id(p) for p in critic}: raise ValueError("Shared parameters require a different update contract")
    if not torch.isfinite(actor_loss+critic_loss): raise ValueError("Nonfinite losses")
    actor_optimizer.zero_grad();critic_optimizer.zero_grad()
    actor_loss.backward();critic_loss.backward()
    actor_norm=torch.nn.utils.clip_grad_norm_(actor,max_norm)
    critic_norm=torch.nn.utils.clip_grad_norm_(critic,max_norm)
    if not torch.isfinite(actor_norm+critic_norm): raise ValueError("Nonfinite gradients")
    actor_optimizer.step();critic_optimizer.step()
    return float(actor_norm),float(critic_norm)

def clipped_value_loss(prediction,old_prediction,target,clip=.2):
    clipped=old_prediction+(prediction-old_prediction).clamp(-clip,clip)
    return .5*torch.maximum((prediction-target).square(),(clipped-target).square()).mean()
