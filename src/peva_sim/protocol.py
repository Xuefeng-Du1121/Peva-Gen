"""Frozen split validation. Model seed must never choose a new data partition."""
import hashlib,json
from pathlib import Path
import numpy as np

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def manifest_indices(dataset, data_path, manifest_path, split):
    m=json.loads(Path(manifest_path).read_text())
    if file_sha(data_path)!=m["sha256"]:
        raise ValueError("Data SHA-256 differs from frozen manifest")
    if set(m["splits"])!={"train","validation","test"}:
        raise ValueError("Expected train/validation/test")
    used=set(); track_sets=[]
    for name,p in m["splits"].items():
        idx=np.asarray(p["indices"],dtype=np.int64)
        if len(idx)==0 or idx.min()<0 or idx.max()>=len(dataset.track_id):
            raise ValueError("Invalid manifest indices")
        if len(set(idx.tolist()))!=len(idx) or used.intersection(idx.tolist()):
            raise ValueError("Overlapping/duplicate split indices")
        used.update(idx.tolist())
        ids=set(dataset.track_id[idx].tolist())
        if ids!=set(p["track_ids"]):
            raise ValueError("Track membership differs from manifest")
        if any(ids & other for other in track_sets):
            raise ValueError("Drifter leakage between splits")
        track_sets.append(ids)
    if len(used)!=len(dataset.track_id):
        raise ValueError("Manifest does not cover dataset")
    return np.asarray(m["splits"][split]["indices"],dtype=np.int64)
