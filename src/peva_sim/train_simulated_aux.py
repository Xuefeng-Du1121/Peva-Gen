"""Simulated multi-target warm-start with recomputed decoder-PVF priorities."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .env import Config
from .latent_state import TargetStateCodec
from .reference_transport import ReferenceTransportJointBelief
from .collect_simulated_belief import TARGET_SCHEMA
from .belief_contract import CONTEXT_SCHEMA
from .particle_snapshot import snapshot_environment,decoded_physics_gradient
from .ocean_training_pool import OceanTrainingPool
from .value_replay import ValueReplaySampler
from .protocol import file_sha
from .reproducibility import snapshot_source
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--codec-steps",type=int,default=1000)
    ap.add_argument("--aux-steps",type=int,default=1000);ap.add_argument("--seed",type=int,default=71)
    a=ap.parse_args()
    source=(ROOT/a.samples).resolve();out=(ROOT/a.out).resolve()
    if not all(p.is_relative_to(ROOT) for p in (source,out)) or min(a.codec_steps,a.aux_steps)<1:
        ap.error("Invalid paths/counts")
    meta=json.loads((source/"metadata.json").read_text())
    if meta["split"]!="train" or meta["target_schema"]!=TARGET_SCHEMA or meta["context_schema"]!=CONTEXT_SCHEMA:
        raise ValueError("Simulated training schema required")
    if file_sha(source/"samples.npz")!=meta["samples_sha256"]: raise ValueError("Samples changed")
    pool=OceanTrainingPool(meta["forcing"]["inventory_path"])
    if pool.inventory_sha256!=meta["forcing"]["inventory_sha256"]: raise ValueError("Inventory changed")
    allowed={r["id"] for r in pool.training_scenarios}
    if any(row["scenario_id"] not in allowed for row in meta["episodes"]): raise ValueError("Non-training scenario")
    if pool.config_dict()!=meta["config"]: raise ValueError("Configuration changed")
    pool.close()
    with np.load(source/"samples.npz") as archive: data={k:archive[k].copy() for k in archive.files}
    if not all(np.isfinite(v).all() for v in data.values()): raise ValueError("Nonfinite source")
    c=Config(**meta["config"])
    if data["target"].shape!=(len(data["context"]),2*c.n_targets): raise ValueError("Target contract mismatch")
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(a.seed);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(a.seed)
    targets=torch.tensor(data["target"],dtype=torch.float32)
    codec=TargetStateCodec(c.n_targets)
    copt=torch.optim.Adam(codec.parameters(),lr=3e-4)
    for step in range(a.codec_steps):
        x=targets[rng.integers(len(targets),size=256)]
        loss=(codec.decode(codec.encode_target(x))-x).square().mean()
        if not torch.isfinite(loss): raise ValueError("Nonfinite codec loss")
        copt.zero_grad();loss.backward();copt.step()
    codec.freeze()
    with torch.no_grad():
        z=codec.encode_target(targets);center=z.mean(0);delta=(z-center).double()
        covariance=delta.T@delta/len(z)
        reconstruction_m=(codec.decode(z)-targets).reshape(-1,c.n_targets,2).norm(dim=-1)*(c.region_m/2)
    priorities=[]
    for i in range(len(targets)):
        snapshot=snapshot_environment(c,data["particles"][i],data["weights"][i],data["found"][i],data["time_s"][i])
        gradient=decoded_physics_gradient(codec,z[i:i+1],snapshot)
        priorities.append(float(gradient.norm().clamp(.1,10)))
    np.save(out/"decoder-priorities.npy",np.asarray(priorities))
    sampler=ValueReplaySampler(priorities)
    model=ReferenceTransportJointBelief(data["context"].shape[-1],codec,center,covariance)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    history=[]
    for step in range(a.aux_steps):
        idx,correction=sampler.sample(rng,32)
        context=torch.tensor(data["context"][idx],dtype=torch.float32)
        losses=model.loss(context,targets[idx],torch.zeros(context.shape[:2]),torch.tensor(data["received"][idx]),
                          torch.tensor(correction,dtype=torch.float32))
        loss=model.objective(losses)
        if not torch.isfinite(loss): raise ValueError("Nonfinite auxiliary loss")
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0 or step==a.aux_steps-1:
            row=dict(step=step+1,**{k:float(losses[k].detach()) for k in ("belief","relevance","rate")})
            history.append(row);print(json.dumps(row),flush=True)
    metadata=dict(architecture=model.architecture,n_targets=c.n_targets,latent_dim=128,context_dim=model.context_dim,
                  context_schema=CONTEXT_SCHEMA,input_target_schema=TARGET_SCHEMA,source_metadata=meta,args=vars(a),
                  codec_training_error_m=float(reconstruction_m.mean()),evaluation="training reconstruction only; no held-out quality claim",
                  priority="recomputed decoder pullback at decoded locations from float64 particle snapshots",
                  priority_sha256=file_sha(out/"decoder-priorities.npy"),priority_min=min(priorities),priority_max=max(priorities),
                  status="simulated multi-target auxiliary warm-start; not full PEVA-Gen",history=history,
                  source_snapshot=snapshot_source(out))
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),codec_optimizer=copt.state_dict(),
                    metadata=metadata,torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state),out/"checkpoint.pt")
    metadata["checkpoint_sha256"]=file_sha(out/"checkpoint.pt")
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(dict(complete=True,n_targets=c.n_targets,training_reconstruction_m=float(reconstruction_m.mean()))),flush=True)

if __name__=="__main__": main()
