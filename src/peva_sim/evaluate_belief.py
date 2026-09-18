"""Belief quality gate on stored rollout contexts; never optimizes model weights."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .joint_transport import TransportJointBelief,quantize_message,aggregate_messages
from .protocol import file_sha
ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True);ap.add_argument("--samples",required=True)
    ap.add_argument("--out",required=True);ap.add_argument("--seed",type=int,default=912)
    ap.add_argument("--draws",type=int,default=32);a=ap.parse_args()
    if a.draws<2: ap.error("at least two samples for energy score")
    cp=(ROOT/a.checkpoint).resolve();source=(ROOT/a.samples).resolve();out=(ROOT/a.out).resolve()
    if not all(p.is_relative_to(ROOT) for p in (cp,source,out)): ap.error("paths outside project")
    if out.exists(): ap.error("refusing overwrite")
    ck=torch.load(cp,map_location="cpu",weights_only=False);meta=ck["metadata"]
    smeta=json.loads((source/"metadata.json").read_text())
    if file_sha(source/"samples.npz")!=smeta["samples_sha256"]: raise ValueError("sample hash mismatch")
    original=meta["source_metadata"]
    for key in ("data_sha256","manifest_sha256","context_schema","target_schema"):
        if original[key]!=smeta[key]: raise ValueError("source/schema mismatch: "+key)
    if original["config"]!=smeta["config"]: raise ValueError("environment mismatch")
    torch.set_num_threads(1)
    model=TransportJointBelief(meta["context_dim"],diffusion_parameterization=meta.get("diffusion_parameterization","epsilon"))
    model.load_state_dict(ck["model"]);model.eval()
    rng=torch.Generator().manual_seed(a.seed)
    with np.load(source/"samples.npz") as f: data={k:f[k].copy() for k in f.files}
    results={k:[] for k in ("mean_error_m","energy_score_m","out_of_region","max_abs_latent","datum_error_m")}
    scale=smeta["config"]["region_m"]/2
    for start in range(0,len(data["target"]),32):
        x=torch.tensor(data["context"][start:start+32]);b,n,d=x.shape
        target=data["target"][start:start+32]
        with torch.no_grad():
            flat=x.reshape(-1,d)
            mu=model.encoder.mu(flat);lv=model.encoder.logvar(flat).clamp(-8,8)
            msg=mu+(.5*lv).exp()*torch.randn(mu.shape,generator=rng)
            mask=torch.tensor(data["received"][start:start+32])
            agg=aggregate_messages(quantize_message(msg).reshape(b,n,32),torch.zeros(b,n),mask)
            condition=torch.cat((x,agg),-1).reshape(b*n,-1)
            initial=torch.randn(b*n,a.draws,2,generator=rng)
            z=model.belief.sample(condition,samples=a.draws,initial_noise=initial).reshape(b,n,a.draws,2).numpy()
        mean_error=np.linalg.norm((z.mean(2)-target[:,None,:])*scale,axis=-1)
        first=np.linalg.norm((z-target[:,None,None,:])*scale,axis=-1).mean(-1)
        # Non-self paired draws are independent samples conditional on context.
        second=np.linalg.norm((z-np.roll(z,1,axis=2))*scale,axis=-1).mean(-1)
        results["mean_error_m"].extend(mean_error.mean(1).tolist())
        results["energy_score_m"].extend((first-.5*second).mean(1).tolist())
        results["out_of_region"].extend((np.abs(z)>1).any(-1).mean((1,2)).tolist())
        results["max_abs_latent"].extend(np.abs(z).max((1,2,3)).tolist())
        results["datum_error_m"].extend(np.linalg.norm(target*scale,axis=-1).tolist())
    report={"checkpoint_sha256":file_sha(cp),"sample_sha256":smeta["samples_sha256"],
            "source_sha256":file_sha(__file__),"split":smeta["split"],"rows":len(data["target"]),
            "draws":a.draws,"seed":a.seed,"parameterization":meta.get("diffusion_parameterization","epsilon"),
            "scope":"auxiliary quality diagnostic on correlated rollout rows, not SAR effectiveness",
            "means":{k:float(np.mean(v)) for k,v in results.items()},
            "max_abs_latent":float(max(results["max_abs_latent"])),
            "episode_ids":data["episode"].tolist(),"per_row":results}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ("split","rows","parameterization","means","max_abs_latent")}),flush=True)

if __name__=="__main__": main()
