"""Train-only codec fitting followed by frozen-codec 128-D auxiliary fitting.

Development on recorded train tracks, NOT the manuscript's simulated-training
protocol. No policy optimization and no claim of complete PEVA-Gen.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .latent_state import TargetStateCodec
from .reference_transport import ReferenceTransportJointBelief as LatentTransportJointBelief
from .belief_contract import CONTEXT_SCHEMA,TARGET_SCHEMA
from .protocol import file_sha,manifest_indices
from .real_dataset import RealDriftDataset
from .value_replay import ValueReplaySampler
from .reproducibility import snapshot_source
ROOT=Path(__file__).resolve().parents[2]

def load_samples(path,split):
    meta=json.loads((path/"metadata.json").read_text())
    if meta["split"]!=split or meta["context_schema"]!=CONTEXT_SCHEMA or meta["target_schema"]!=TARGET_SCHEMA:
        raise ValueError("Sample schema/split mismatch")
    raw=ROOT/"runs/real-east-china-2018-01-utc-v2/segments.npz"
    manifest=raw.parent/"split-manifest.json"
    for key,p in (("samples_sha256",path/"samples.npz"),("data_sha256",raw),("manifest_sha256",manifest)):
        if meta[key]!=file_sha(p): raise ValueError("Source hash mismatch")
    with np.load(path/"samples.npz") as archive:
        data={k:archive[k].copy() for k in archive.files}
    if not all(np.isfinite(v).all() for v in data.values()): raise ValueError("Nonfinite data")
    allowed=manifest_indices(RealDriftDataset(raw),raw,manifest,split)
    if not np.isin(data["global_index"],allowed).all(): raise ValueError("Split leakage")
    if data["target"].shape!=(len(data["context"]),2): raise ValueError("One-target development data expected")
    return data,meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples",required=True);ap.add_argument("--validation",required=True)
    ap.add_argument("--out",required=True);ap.add_argument("--codec-steps",type=int,default=1000)
    ap.add_argument("--aux-steps",type=int,default=1000);ap.add_argument("--seed",type=int,default=71)
    a=ap.parse_args()
    if min(a.codec_steps,a.aux_steps)<1: ap.error("Positive steps required")
    paths=[(ROOT/v).resolve() for v in (a.samples,a.validation,a.out)]
    if not all(p.is_relative_to(ROOT) for p in paths): ap.error("Paths outside project")
    source,validation,out=paths
    data,meta=load_samples(source,"train")
    val,vmeta=load_samples(validation,"validation")
    if np.intersect1d(data["global_index"],val["global_index"]).size: raise ValueError("Split overlap")
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(a.seed);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(a.seed)
    codec=TargetStateCodec()
    optimizer=torch.optim.Adam(codec.parameters(),lr=3e-4)
    target=torch.tensor(data["target"],dtype=torch.float32)
    for step in range(a.codec_steps):
        x=target[rng.integers(len(target),size=256)]
        loss=(codec.decode(codec.encode_target(x))-x).square().mean()
        if not torch.isfinite(loss): raise ValueError("Nonfinite reconstruction loss")
        optimizer.zero_grad();loss.backward();optimizer.step()
    codec.freeze()
    with torch.no_grad():
        vt=torch.tensor(val["target"],dtype=torch.float32)
        error=(codec.decode(codec.encode_target(vt))-vt).norm(dim=-1)*(meta["config"]["region_m"]/2)
        quality=dict(validation_rows=len(vt),codec_mean_error_m=float(error.mean()),
                     codec_max_error_m=float(error.max()),scope="decoder reconstruction, not belief or search quality")
    with torch.no_grad():
        latent_training=codec.encode_target(target)
        reference_mean=latent_training.mean(0)
        centered=latent_training-reference_mean
        reference_covariance=centered.double().T@centered.double()/len(centered)
    model=LatentTransportJointBelief(data["context"].shape[-1],codec,reference_mean,reference_covariance)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    sampler=ValueReplaySampler(data["priority"])
    history=[]
    for step in range(a.aux_steps):
        idx,correction=sampler.sample(rng,32)
        context=torch.tensor(data["context"][idx],dtype=torch.float32)
        losses=model.loss(context,target[idx],torch.zeros(context.shape[:2]),
                          torch.tensor(data["received"][idx]),torch.tensor(correction,dtype=torch.float32))
        total=model.objective(losses)
        if not torch.isfinite(total): raise ValueError("Nonfinite latent auxiliary loss")
        opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0 or step==a.aux_steps-1:
            row=dict(step=step+1,**{k:float(losses[k].detach()) for k in ("belief","relevance","rate")})
            history.append(row);print(json.dumps(row),flush=True)
    metadata=dict(architecture=model.architecture,args=vars(a),source_metadata=meta,
                  validation_metadata=vmeta,context_dim=model.context_dim,latent_dim=128,n_targets=1,
                  context_schema=CONTEXT_SCHEMA,input_target_schema=TARGET_SCHEMA,
                  status="Gaussian-reference 128-D auxiliary development; not full PEVA-Gen",
                  reference="frozen training-only Gaussian moments plus learned v residual; unchanged epsilon loss",
                  codec_training="uniform training rows; fixed steps; frozen for all denoising updates",
                  replay_priority="inherited spatial gradient prototype; NOT yet neural-decoder pullback priority",
                  quality=quality,history=history,source_snapshot=snapshot_source(out))
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),codec_optimizer=optimizer.state_dict(),
                    metadata=metadata,torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state),out/"checkpoint.pt")
    metadata["checkpoint_sha256"]=file_sha(out/"checkpoint.pt")
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(quality),flush=True)

if __name__=="__main__": main()
