"""Read-only latent posterior diagnostics on fixed validation rows; no tuning."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .train_latent_aux import load_samples
from .latent_state import TargetStateCodec
from .reference_transport import ReferenceTransportJointBelief as LatentTransportJointBelief
from .joint_transport import aggregate_messages,quantize_message
from .protocol import file_sha
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args()
    cp=(ROOT/a.checkpoint).resolve();out=(ROOT/a.out).resolve()
    if not all(p.is_relative_to(ROOT) for p in (cp,out)): ap.error("Outside project")
    if out.exists(): ap.error("Refusing overwrite")
    torch.set_num_threads(1)
    ck=torch.load(cp,map_location="cpu",weights_only=False);meta=ck["metadata"]
    if meta.get("architecture")!=LatentTransportJointBelief.architecture: raise ValueError("Architecture mismatch")
    train,tm=load_samples(ROOT/meta["args"]["samples"],"train")
    val,vm=load_samples(ROOT/meta["args"]["validation"],"validation")
    if tm["samples_sha256"]!=meta["source_metadata"]["samples_sha256"]: raise ValueError("Training samples changed")
    if vm["samples_sha256"]!=meta["validation_metadata"]["samples_sha256"]: raise ValueError("Validation samples changed")
    model=LatentTransportJointBelief(meta["context_dim"],TargetStateCodec())
    model.load_state_dict(ck["model"]);model.eval()
    rng=torch.Generator().manual_seed(730113)
    indices=np.linspace(0,len(val["target"])-1,min(128,len(val["target"])),dtype=int)
    context=torch.tensor(val["context"][indices],dtype=torch.float32)
    received=torch.tensor(val["received"][indices])
    target=torch.tensor(val["target"][indices],dtype=torch.float32)
    b,n,d=context.shape
    with torch.no_grad():
        train_z=model.codec.encode_target(torch.tensor(train["target"],dtype=torch.float32))
        center=train_z.mean(0)
        covariance=(train_z-center).T@(train_z-center)/len(train_z)
        eig,vec=torch.linalg.eigh(covariance)
        basis=vec[:,-2:]
        mu=model.encoder.mu(context.reshape(b*n,d))
        lv=model.encoder.logvar(context.reshape(b*n,d)).clamp(-8,8)
        messages=mu+(.5*lv).exp()*torch.randn(mu.shape,generator=rng)
        aggregate=aggregate_messages(quantize_message(messages).reshape(b,n,32),torch.zeros(b,n),received)
        aggregate=aggregate.reshape(b*n,32)
        local=context.reshape(b*n,d)
        targets=target[:,None].expand(b,n,2).reshape(b*n,2)
        truth_z=model.codec.encode_target(targets)
        noise=torch.randn((b*n,16,128),generator=rng)
        rows=[]
        residual_state={k:v.clone() for k,v in model.belief.eps.residual.state_dict().items()}
        for steps in (5,20):
            model.belief.steps=steps
            for conditioning in ("aligned","row-shuffled","reference-only"):
                model.belief.eps.residual.load_state_dict(residual_state)
                if conditioning=="reference-only":
                    model.belief.eps.residual.net[-1].weight.zero_()
                    model.belief.eps.residual.net[-1].bias.zero_()
                order=torch.arange(b).roll(1) if conditioning=="row-shuffled" else torch.arange(b)
                x=local.reshape(b,n,d)[order].reshape(b*n,d)
                agg=aggregate.reshape(b,n,32)[order].reshape(b*n,32)
                z,xy=model.sample(x,agg,samples=16,initial_noise=noise)
                zmean=z.mean(1);deviation=z-zmean[:,None]
                spread=deviation.square().sum(-1).mean(1)
                orth=deviation-(deviation@basis)@basis.T
                scale=tm["config"]["region_m"]/2
                point_error=(xy.mean(1)-targets).norm(dim=-1)*scale
                radius=(xy-xy.mean(1)[:,None]).norm(dim=-1)*scale
                first=(xy-targets[:,None]).norm(dim=-1).mean(1)
                pair=(xy[:,:,None]-xy[:,None,:]).norm(dim=-1).mean((1,2))
                rows.append(dict(steps=steps,conditioning=conditioning,
                    latent_spread=float(spread.mean()),
                    latent_error_sq=float((zmean-truth_z).square().sum(-1).mean()),
                    spread_outside_training_top2_pc=float(orth.square().sum(-1).mean()),
                    physical_mean_error_m=float(point_error.mean()),
                    physical_rms_radius_m=float(radius.square().mean().sqrt()),
                    empirical_q90_ball_coverage=float((point_error<=radius.quantile(.9,dim=1)).float().mean()),
                    physical_energy_score_m=float(((first-.5*pair)*scale).mean()),
                    latent_risk_gate=float((2*torch.sigmoid(-2*spread.double())).mean()),
                    out_of_region_fraction=float((xy.abs()>1).any(-1).float().mean())))
    report=dict(status="diagnostic only",checkpoint_sha256=file_sha(cp),source_sha256=file_sha(__file__),
                validation_samples_sha256=vm["samples_sha256"],validation_indices=indices.tolist(),
                sampling_seed=730113,draws=16,train_latent_variance_trace=float(eig.sum()),
                train_top2_pc_fraction=float(eig[-2:].sum()/eig.sum()),
                datum_mean_error_m=float(target.norm(dim=-1).mean()*scale),rows=rows,
                caveat="Correlated validation rows; global PCA is a diagnostic approximation, not proof of a linear state manifold. K=20 is diagnostic, not selected deployment setting.")
    with out.open("x") as f: json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)

if __name__=="__main__": main()
