"""128-D packet transport adapter; explicit architecture and frozen codec."""
from pathlib import Path
import numpy as np
import torch
from .belief_contract import CONTEXT_SCHEMA,TARGET_SCHEMA,swarm_context
from .communication import encode_message,decode_message,aggregate_inbox
from .joint_transport import quantize_message
from .latent_state import TargetStateCodec
from .reference_transport import ReferenceTransportJointBelief as LatentTransportJointBelief
from .message_relevance import public_grid_gradient
from .protocol import file_sha

class LatentFeatureAdapter:
    feature_dim=176
    def __init__(self,checkpoint,config,seed=0,samples=2,message_weighting="uniform"):
        if config.n_targets!=1 or samples<1 or message_weighting not in ("uniform","physics_grid"):
            raise ValueError("Unsupported target/sample/message configuration")
        self.config=config;self.samples=samples;self.message_weighting=message_weighting
        self.checkpoint_sha256=file_sha(checkpoint)
        ck=torch.load(Path(checkpoint),map_location="cpu",weights_only=False)
        self.metadata=ck["metadata"]
        if self.metadata.get("architecture")!=LatentTransportJointBelief.architecture:
            raise ValueError("128-D architecture required; legacy checkpoint forbidden")
        if self.metadata["context_schema"]!=CONTEXT_SCHEMA or self.metadata["input_target_schema"]!=TARGET_SCHEMA:
            raise ValueError("Context/target schema mismatch")
        if self.metadata["context_dim"]!=16+config.grid_side**2:
            raise ValueError("Grid/context mismatch")
        source=self.metadata["source_metadata"]["config"]
        for key in ("region_m","horizon_s","coverage_mode"):
            if source[key]!=getattr(config,key): raise ValueError("Source normalization/config mismatch")
        self.model=LatentTransportJointBelief(self.metadata["context_dim"],TargetStateCodec())
        self.model.load_state_dict(ck["model"]);self.model.eval();self.model.requires_grad_(False)
        self.rng=torch.Generator(device="cpu").manual_seed(seed)
    @torch.no_grad()
    def step(self,env,observation):
        context=swarm_context(observation,self.config)
        x=torch.tensor(context)
        mu=self.model.encoder.mu(x);lv=self.model.encoder.logvar(x).clamp(-8,8)
        message=mu+(.5*lv).exp()*torch.randn(mu.shape,generator=self.rng)
        scores=np.zeros(len(x))
        if self.message_weighting=="physics_grid":
            z=self.model.decoder(quantize_message(message))
            locations=self.model.codec.decode_locations_m(z,self.config.region_m)
            gx=public_grid_gradient(observation,locations[:,0].numpy())
            unit=1/(2*np.pi*self.config.prior_sigma_m**2)
            g=torch.tensor(gx[:,None,:]/unit,dtype=z.dtype)
            gz=self.model.codec.pullback_location_gradient(z,g,self.config.region_m)
            scores=gz.norm(dim=-1).clamp(.1,10).numpy()
        packets=[encode_message(row,float(np.exp(score))) for row,score in zip(message.numpy(),scores)]
        scores=np.log([decode_message(packet)[1] for packet in packets])
        inbox=env.exchange_messages(packets)
        aggregate=np.stack([aggregate_inbox(items+[(i,packets[i])])[0] for i,items in enumerate(inbox)])
        noise=torch.randn((len(x),self.samples,128),generator=self.rng)
        z,decoded=self.model.sample(x,torch.tensor(aggregate),samples=self.samples,initial_noise=noise)
        mean=z.mean(1);spread=(z-mean[:,None]).square().sum(-1).mean(1)
        features=np.concatenate((context[:,:15],aggregate,mean.numpy(),spread[:,None].numpy()),-1).astype(np.float32)
        if not np.isfinite(features).all(): raise FloatingPointError("Nonfinite latent actor inputs")
        self.last=dict(features=features.copy(),received=inbox,message_scores=scores.copy(),
                       belief_samples=z.numpy(),decoded_samples=decoded.numpy(),spread=spread.numpy())
        return features
