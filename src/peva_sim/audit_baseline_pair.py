"""Audit completed paired baseline runs before any staged evaluation."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import torch
from .protocol import file_sha
ROOT=Path(__file__).resolve().parents[2]
DEPENDENCIES=("env.py","real_env.py","ocean_field.py","real_replay.py","real_dataset.py",
              "protocol.py","ppo_math.py","mappo_v2.py","scenario_manifest.py","evaluate_paired.py")


def audit(pair):
    pair=Path(pair).resolve()
    if not pair.is_relative_to(ROOT): raise ValueError("pair outside project")
    state=json.loads((pair/"status.json").read_text())
    if state["status"]!="completed": raise ValueError("training pair is not terminal-success")
    configurations=[];training_arguments=[];methods={};reference_sources=None
    for method in ("mappo","pvf"):
        run=state["runs"][method]
        if run.get("exit_code")!=0: raise ValueError("nonzero child exit")
        args=run["command"]
        updates=int(args[args.index("--updates")+1])
        rollout=int(args[args.index("--rollout")+1])
        every=int(args[args.index("--snapshot-every")+1])
        expected=[i*rollout for i in range(1,updates+1) if i%every==0 or i==updates]
        directory=pair/method
        metadata=json.loads((directory/"metadata.json").read_text())
        if metadata["args"]["seed"]!=state["seed"] or metadata["args"]["pvf"]!=(method=="pvf"):
            raise ValueError("seed/method mismatch")
        if metadata["config"].get("coverage_mode")!="intensity":
            raise ValueError("not manuscript intensity mode")
        if any(metadata["args"].get(k) for k in ("transport_aux","peva_aux","guidance")):
            raise ValueError("not a plain baseline")
        configurations.append(metadata["config"])
        training_arguments.append({k:v for k,v in metadata["args"].items() if k not in ("out","pvf")})
        if metadata["args"]["updates"]!=updates or metadata["args"]["rollout"]!=rollout:
            raise ValueError("launch command/checkpoint budget mismatch")
        for key,digest_key in (("data","data_sha256"),("manifest","manifest_sha256")):
            source=(ROOT/metadata["args"][key]).resolve()
            if not source.is_relative_to(ROOT) or file_sha(source)!=metadata[digest_key]:
                raise ValueError("raw data/split provenance mismatch")
        manifest=json.loads((directory/"source-manifest.json").read_text())
        digest=file_sha(directory/"source.tar.gz")
        if digest!=manifest["archive_sha256"] or digest!=metadata["source_archive_sha256"]:
            raise ValueError("source bundle hash mismatch")
        with tarfile.open(directory/"source.tar.gz") as tar:
            for name,digest in manifest["files"].items():
                stream=tar.extractfile(name)
                if stream is None or hashlib.sha256(stream.read()).hexdigest()!=digest:
                    raise ValueError("archived source mismatch: "+name)
        sources={name:manifest["files"]["src/peva_sim/"+name] for name in DEPENDENCIES}
        if reference_sources is not None and sources!=reference_sources:
            raise ValueError("paired runs used different baseline sources")
        reference_sources=sources
        for name,digest in sources.items():
            if file_sha(ROOT/"src/peva_sim"/name)!=digest:
                raise ValueError("current evaluator dependency changed: "+name)
        found=sorted(int(p.stem.split("-")[-1]) for p in directory.glob("checkpoint-step-*.pt"))
        if found!=expected: raise ValueError("missing/unexpected stage checkpoints")
        checkpoints=[]
        for steps in expected:
            path=directory/f"checkpoint-step-{steps:09d}.pt"
            ck=torch.load(path,map_location="cpu",weights_only=False)
            if ck["environment_steps"]!=steps or ck["update"]+1!=steps//rollout:
                raise ValueError("checkpoint filename/content mismatch")
            if ck["metadata"]!=metadata: raise ValueError("checkpoint metadata mismatch")
            checkpoints.append({"steps":steps,"path":str(path),"sha256":file_sha(path)})
        final=torch.load(directory/"checkpoint.pt",map_location="cpu",weights_only=False)
        if final["environment_steps"]!=state["budget_steps_per_run"]:
            raise ValueError("final budget incomplete")
        if not all(torch.equal(final["model"][k],v) for k,v in ck["model"].items()):
            raise ValueError("latest and final-stage model differ")
        methods[method]={"checkpoints":checkpoints,"source_archive_sha256":manifest["archive_sha256"]}
    if configurations[0]!=configurations[1]: raise ValueError("unequal simulator configurations")
    if training_arguments[0]!=training_arguments[1]: raise ValueError("unequal training arguments")
    return {"status":"verified","pair":str(pair),"budget_steps":state["budget_steps_per_run"],
            "seed":state["seed"],"methods":methods,
            "scope":"completion, source and checkpoint integrity; not convergence/performance"}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--pair",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args();out=(ROOT/a.out).resolve()
    if not out.is_relative_to(ROOT) or out.exists(): ap.error("output outside project or already exists")
    result=audit(ROOT/a.pair);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({"status":result["status"],"budget_steps":result["budget_steps"]}),flush=True)
if __name__=="__main__": main()
