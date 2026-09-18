"""Frozen transport-aware auxiliary rollout adapter (not full PEVA-Gen)."""
from pathlib import Path
import numpy as np
import torch
from .belief_contract import CONTEXT_SCHEMA,TARGET_SCHEMA,swarm_context
from .communication import encode_message,decode_message,aggregate_inbox
from .joint_transport import quantize_message
from .joint_transport import TransportJointBelief
from .protocol import file_sha


class TransportFeatureAdapter:
    feature_dim=50  # own operational/contact features(15), message(32), mean(2), spread(1)

    def __init__(self,checkpoint,config,seed=0,samples=4,message_weighting="uniform"):
        if message_weighting not in ("uniform","physics_grid"):
            raise ValueError("unsupported message weighting")
        self.message_weighting=message_weighting
        if config.n_targets!=1 or samples<1:
            raise ValueError("single-target adapter and positive sample count required")
        self.config=config;self.samples=samples
        self.checkpoint_sha256=file_sha(checkpoint)
        ck=torch.load(Path(checkpoint),map_location="cpu",weights_only=False)
        self.metadata=ck["metadata"]
        if self.metadata["context_schema"]!=CONTEXT_SCHEMA or self.metadata["target_schema"]!=TARGET_SCHEMA:
            raise ValueError("unsupported auxiliary contract")
        if self.metadata["context_dim"]!=16+config.grid_side**2:
            raise ValueError("grid/context mismatch")
        source_config=self.metadata["source_metadata"]["config"]
        if source_config.get("coverage_mode","conditional") != config.coverage_mode:
            raise ValueError("coverage mode differs from auxiliary training")
        for key in ("region_m","horizon_s"):
            if source_config[key]!=getattr(config,key):
                raise ValueError("auxiliary coordinate/time normalization mismatch")
        self.model=TransportJointBelief(self.metadata["context_dim"],
                        diffusion_parameterization=self.metadata.get("diffusion_parameterization","epsilon"))
        self.model.load_state_dict(ck["model"]);self.model.eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        self.rng=torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def step(self,env,observation):
        context=swarm_context(observation,self.config)
        x=torch.tensor(context)
        # Sample variational messages using a dedicated stream, as in training.
        mu=self.model.encoder.mu(x)
        logvar=self.model.encoder.logvar(x).clamp(-8,8)
        message=mu+(.5*logvar).exp()*torch.randn(mu.shape,generator=self.rng)
        scores=np.zeros(len(x))
        if self.message_weighting=="physics_grid":
            from .message_relevance import physics_message_scores
            estimated=self.model.decoder(quantize_message(message)).numpy()
            scores=physics_message_scores(observation,estimated,self.config)
        packets=[encode_message(row,float(np.exp(score))) for row,score in zip(message.numpy(),scores)]
        # Cache the float32 relevance actually carried by the packet.
        scores=np.log([decode_message(packet)[1] for packet in packets])
        inbox=env.exchange_messages(packets)
        aggregates=np.stack([aggregate_inbox(items+[(i,packets[i])])[0]
                             for i,items in enumerate(inbox)])
        condition=torch.cat((x,torch.tensor(aggregates)),-1)
        noise=torch.randn((len(x),self.samples,2),generator=self.rng)
        belief=self.model.belief.sample(condition,samples=self.samples,initial_noise=noise)
        mean=belief.mean(1)
        spread=(belief-mean[:,None,:]).square().sum(-1).mean(1)
        features=np.concatenate((context[:,:15],aggregates,mean.numpy(),spread[:,None].numpy()),axis=1).astype(np.float32)
        if not np.isfinite(features).all(): raise FloatingPointError("nonfinite auxiliary features")
        self.last={"features":features.copy(),"received":inbox,
                   "message_scores":scores.copy(),
                   "belief_samples":belief.numpy(),"spread":spread.numpy()}
        return features
