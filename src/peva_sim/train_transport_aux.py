"""Reproducible physics-prioritized auxiliary warm-start; not full PEVA-Gen."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from .belief_contract import CONTEXT_SCHEMA,TARGET_SCHEMA
from .joint_transport import TransportJointBelief
from .value_replay import ValueReplaySampler
from .protocol import file_sha
ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--seed",type=int,default=71)
    ap.add_argument("--diffusion-parameterization",choices=("epsilon","v_residual"),default="v_residual")
    a=ap.parse_args()
    if min(a.steps,a.batch)<1: ap.error("positive training counts required")
    source=(ROOT/a.samples).resolve(); out=(ROOT/a.out).resolve()
    if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT):
        ap.error("paths must remain within project")
    meta=json.loads((source/"metadata.json").read_text())
    if meta["context_schema"]!=CONTEXT_SCHEMA or meta["target_schema"]!=TARGET_SCHEMA or meta["split"]!="train":
        raise ValueError("incompatible schema or non-training source")
    if meta["samples_sha256"]!=file_sha(source/"samples.npz"):
        raise ValueError("sample hash mismatch")
    with np.load(source/"samples.npz") as archive:
        data={k:archive[k].copy() for k in archive.files}
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("nonfinite training data")
    # Revalidate provenance against the frozen source split, not only its label.
    from .real_dataset import RealDriftDataset
    from .protocol import manifest_indices
    raw=ROOT/"runs/real-east-china-2018-01-utc-v2/segments.npz"
    manifest=raw.parent/"split-manifest.json"
    if meta["data_sha256"]!=file_sha(raw) or meta["manifest_sha256"]!=file_sha(manifest):
        raise ValueError("raw source or split changed")
    allowed=manifest_indices(RealDriftDataset(raw),raw,manifest,"train")
    if not np.isin(data["global_index"],allowed).all():
        raise ValueError("held-out rows in training samples")
    out.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed);torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(a.seed)
    model=TransportJointBelief(data["context"].shape[-1],diffusion_parameterization=a.diffusion_parameterization)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    sampler=ValueReplaySampler(data["priority"])
    history=[]
    for step in range(a.steps):
        idx,correction=sampler.sample(rng,a.batch)
        context=torch.tensor(data["context"][idx])
        target=torch.tensor(data["target"][idx])
        received=torch.tensor(data["received"][idx])
        # Uniform aggregation is only an auxiliary initialization. No privileged
        # target-dependent relevance is supplied to the deployment message path.
        scores=torch.zeros(context.shape[:2])
        losses=model.loss(context,target,scores,received,torch.tensor(correction,dtype=torch.float32))
        total=model.objective(losses)
        if not torch.isfinite(total): raise FloatingPointError("nonfinite auxiliary loss")
        opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0 or step==a.steps-1:
            row={"step":step+1,**{k:float(losses[k].detach()) for k in ("belief","relevance","rate")}}
            history.append(row);print(json.dumps(row),flush=True)
    metadata={"status":"auxiliary warm-start, no held-out quality evaluation; not full PEVA-Gen",
              "context_schema":CONTEXT_SCHEMA,"target_schema":TARGET_SCHEMA,
              "diffusion_parameterization":a.diffusion_parameterization,
              "context_dim":model.context_dim,"args":vars(a),"source_metadata":meta,
              "sample_metadata_sha256":file_sha(source/"metadata.json"),
              "aggregation":"uniform among delivered neighbors and self",
              "priority":meta.get("priority_schema","legacy-current-peak-normalization"),
              "sampling":"p(i)=priority[i]/sum(priority); rate correction=1/(N*p(i)); no extra relevance/denoising weight",
              "history":history,"source_hashes":{p:file_sha(ROOT/"src/peva_sim"/p) for p in
                  ("train_transport_aux.py","joint_transport.py","belief_contract.py","value_replay.py",
                   "physics_priority.py","diffusion_belief.py","residual_diffusion.py","peva_gen.py","communication.py")}}
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"metadata":metadata,
                "torch_rng":torch.get_rng_state(),"numpy_rng":rng.bit_generator.state,
                "steps":a.steps},out/"checkpoint.pt")
    metadata["checkpoint_sha256"]=file_sha(out/"checkpoint.pt")
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps({"complete":True,"out":str(out)}),flush=True)

if __name__=="__main__": main()
