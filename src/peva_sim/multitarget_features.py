"""Belief/communication adapter with observable operational reach features."""
from pathlib import Path
import numpy as np
import torch
from .belief_contract import CONTEXT_SCHEMA,swarm_context
from .collect_simulated_belief import TARGET_SCHEMA
from .communication import encode_message,decode_message,aggregate_inbox
from .joint_transport import quantize_message
from .latent_state import TargetStateCodec
from .reference_transport import ReferenceTransportJointBelief as LatentTransportJointBelief
from .public_multitarget_value import public_multitarget_gradient
from .protocol import file_sha


def belief_operational_features(decoded_samples, positions_m, config):
    """Build relative candidate locations and reach times from belief only."""
    decoded = np.asarray(decoded_samples, dtype=np.float32)
    positions = np.asarray(positions_m, dtype=np.float32)
    if decoded.ndim != 3 or decoded.shape != (
            config.n_uavs, decoded.shape[1], 2 * config.n_targets):
        raise ValueError("Decoded belief shape does not match UAV/target contract")
    if positions.shape != (config.n_uavs, 2):
        raise ValueError("Position shape does not match UAV contract")
    if not np.isfinite(decoded).all() or not np.isfinite(positions).all():
        raise ValueError("Operational belief features require finite inputs")
    locations = ((decoded + 1.) * (config.region_m / 2.)).reshape(
        config.n_uavs, decoded.shape[1], config.n_targets, 2)
    locations = np.clip(locations, 0., config.region_m).mean(axis=1)
    relative = (locations - positions[:, None, :]) / config.region_m
    reach = np.linalg.norm(
        locations - positions[:, None, :], axis=-1) / (
            config.speed_mps * config.horizon_s)
    return np.concatenate((
        relative.reshape(config.n_uavs, -1), reach), axis=-1).astype(np.float32)


class MultiTargetFeatureAdapter:
    def __init__(self,checkpoint,config,seed=0,samples=2,message_weighting="uniform",
                 device="cpu",deterministic=False,physics_context=True):
        if config.n_targets<1 or samples<1 or message_weighting not in ("uniform","physics_grid"):
            raise ValueError("Unsupported target/sample/message configuration")
        if not isinstance(deterministic,bool) or not isinstance(physics_context,bool):
            raise ValueError("Ablation switches must be boolean")
        self.device=torch.device(device)
        if self.device.type=="cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable")
        self.config=config;self.samples=samples;self.message_weighting=message_weighting
        self.base_feature_dim=176+3*config.n_targets
        self.feature_dim=self.base_feature_dim+config.n_uavs
        self.deterministic=deterministic;self.physics_context=physics_context
        self.checkpoint_sha256=file_sha(checkpoint)
        ck=torch.load(Path(checkpoint),map_location="cpu",weights_only=False)
        self.metadata=ck["metadata"]
        if self.metadata.get("n_targets")!=config.n_targets:
            raise ValueError("Auxiliary target count mismatch")
        if self.metadata.get("architecture")!=LatentTransportJointBelief.architecture:
            raise ValueError("128-D architecture required; legacy checkpoint forbidden")
        if self.metadata["context_schema"]!=CONTEXT_SCHEMA or self.metadata["input_target_schema"]!=TARGET_SCHEMA:
            raise ValueError("Context/target schema mismatch")
        if self.metadata["context_dim"]!=16+config.grid_side**2:
            raise ValueError("Grid/context mismatch")
        source=self.metadata["source_metadata"]["config"]
        for key in ("region_m","horizon_s","coverage_mode"):
            if source[key]!=getattr(config,key): raise ValueError("Source normalization/config mismatch")
        self.model=LatentTransportJointBelief(self.metadata["context_dim"],TargetStateCodec(config.n_targets))
        self.model.load_state_dict(ck["model"]);self.model.to(self.device);self.model.eval();self.model.requires_grad_(False)
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
    @torch.no_grad()
    def step(self,env,observation):
        if self.message_weighting=="physics_grid" and "confirmed_targets" not in observation:
            raise ValueError("Physics messages require delivered confirmation protocol")
        context=swarm_context(observation,self.config)
        if not self.physics_context:
            context[:,15:]=0
        x=torch.as_tensor(context,device=self.device)
        mu=self.model.encoder.mu(x);lv=self.model.encoder.logvar(x).clamp(-8,8)
        message=(mu if self.deterministic else
                 mu+(.5*lv).exp()*torch.randn(
                     mu.shape,generator=self.rng,device=self.device))
        scores=np.zeros(len(x))
        if self.message_weighting=="physics_grid":
            z=self.model.decoder(quantize_message(message))
            gz=public_multitarget_gradient(self.model.codec,z,observation,
                                           observation["confirmed_targets"],self.config)
            scores=gz.norm(dim=-1).clamp(.1,10).cpu().numpy()
        packets=[encode_message(row,float(np.exp(score))) for row,score in zip(message.cpu().numpy(),scores)]
        scores=np.log([decode_message(packet)[1] for packet in packets])
        inbox=env.exchange_messages(packets)
        aggregate=np.stack([aggregate_inbox(items+[(i,packets[i])])[0] for i,items in enumerate(inbox)])
        noise=(torch.zeros((len(x),self.samples,128),device=self.device)
               if self.deterministic else torch.randn(
                   (len(x),self.samples,128),generator=self.rng,device=self.device))
        z,decoded=self.model.sample(x,torch.as_tensor(aggregate,device=self.device),samples=self.samples,initial_noise=noise)
        mean=z.mean(1);spread=(z-mean[:,None]).square().sum(-1).mean(1)
        decoded_np=decoded.cpu().numpy()
        operational=belief_operational_features(
            decoded_np, observation["positions_m"], self.config)
        roles=np.eye(self.config.n_uavs,dtype=np.float32)
        features=np.concatenate((context[:,:15],aggregate,mean.cpu().numpy(),
                                 operational,spread[:,None].cpu().numpy(),roles),
                                -1).astype(np.float32)
        if not np.isfinite(features).all(): raise FloatingPointError("Nonfinite latent actor inputs")
        self.last=dict(features=features.copy(),context=context.copy(),
                       deterministic=self.deterministic,
                       physics_context=self.physics_context,received=inbox,message_scores=scores.copy(),
                       belief_samples=z.cpu().numpy(),decoded_samples=decoded_np,
                       operational_features=operational.copy(),
                       spread=spread.cpu().numpy())
        return features
