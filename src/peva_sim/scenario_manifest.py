"""Outcome-independent, nonoverlapping, track-balanced evaluation episodes."""
import argparse,json
from pathlib import Path
import numpy as np
from .real_dataset import RealDriftDataset
from .protocol import file_sha,manifest_indices

def select_windows(dataset, indices, count):
    groups={}
    for local_index,global_index in enumerate(indices):
        ident=str(dataset.track_id[global_index])
        groups.setdefault(ident,[]).append((int(global_index),local_index))
    candidates={}
    for ident,items in sorted(groups.items()):
        end=-float("inf"); accepted=[]
        for gi,li in sorted(items,key=lambda p:float(dataset.times[p[0],0])):
            t=dataset.times[gi]
            if not np.all(np.diff(t)==3600): raise ValueError("Non-hourly source window")
            if float(t[0])>end:
                accepted.append((gi,li)); end=float(t[-1])
        candidates[ident]=accepted
    if count< len(candidates): raise ValueError("Must represent every held-out track")
    quota={k:0 for k in candidates}
    for _ in range(count):
        possible=[k for k,v in candidates.items() if quota[k]<len(v)]
        if not possible: raise ValueError("Insufficient nonoverlapping windows")
        chosen=min(possible,key=lambda k:(quota[k],k)); quota[chosen]+=1
    result=[]
    for ident,items in candidates.items():
        q=quota[ident]
        if not q: continue
        for position in np.linspace(0,len(items)-1,q,dtype=int):
            gi,li=items[position]
            result.append({"track_id":ident,"global_index":gi,"replay_index":li,
                           "start_utc":float(dataset.times[gi,0]),
                           "end_utc":float(dataset.times[gi,-1])})
    for i,row in enumerate(result):
        row.update(episode_id=f"episode-{i:04d}",environment_seed=810000+i)
    return result

def validate_windows(rows):
    seen=set(); ends={}
    for r in sorted(rows,key=lambda x:(x["track_id"],x["start_utc"])):
        if r["episode_id"] in seen: raise ValueError("Duplicate episode id")
        seen.add(r["episode_id"])
        if r["end_utc"]-r["start_utc"]!=10800: raise ValueError("Wrong horizon")
        if r["start_utc"]<=ends.get(r["track_id"],-float("inf")):
            raise ValueError("Overlapping windows")
        ends[r["track_id"]]=r["end_utc"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True); ap.add_argument("--split-manifest",required=True)
    ap.add_argument("--split",choices=["validation","test"],required=True)
    ap.add_argument("--count",type=int,required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); d=RealDriftDataset(a.data)
    idx=manifest_indices(d,a.data,a.split_manifest,a.split)
    rows=select_windows(d,idx,a.count); validate_windows(rows)
    payload={"version":1,"split":a.split,"data_sha256":file_sha(a.data),
             "split_sha256":file_sha(a.split_manifest),"episodes":rows,
             "selection":"track-balanced, no shared endpoints, no outcomes used",
             "independence":"windows from the same drifter remain correlated"}
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x") as f: json.dump(payload,f,indent=2)
    print(json.dumps({"output":str(p),"count":len(rows),
          "tracks":{k:sum(r["track_id"]==k for r in rows) for k in sorted(set(r["track_id"] for r in rows))}}))
if __name__=="__main__": main()
