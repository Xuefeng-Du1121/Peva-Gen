"""Strict date AND geographic split checks including interpolation support.

Validation concerns environmental support, not unseen trajectory labels or
forecast issue-time validity. Shared boundary grid nodes count as overlap.
"""
import numpy as np
from .real_env import LocalFrame

def support(field,origin_lonlat,start_utc,region_m,horizon_s):
    if region_m<=0 or horizon_s<=0:
        raise ValueError("Positive region and horizon required")
    frame=LocalFrame(origin_lonlat,region_m)
    bounds=frame.inverse(np.array([[0.,0.],[region_m,region_m]]))
    queries=((field.time,np.array([start_utc,start_utc+horizon_s])),
             (field.lat,bounds[:,1]),(field.lon,bounds[:,0]))
    result={}
    for name,(axis,q) in zip(("time","latitude","longitude"),queries):
        idx,_=field._bracket(axis,q)
        lo=int(idx.min());hi=int(idx.max()+1)
        result[name]=[float(axis[lo]),float(axis[hi])]
        result[name+"_indices"]=[lo,hi]
    return result

def intervals_overlap(a,b):
    return max(a[0],b[0])<=min(a[1],b[1])

def validate_split_support(rows):
    if not rows: raise ValueError("Empty scenario inventory")
    ids=set()
    for row in rows:
        if row["id"] in ids: raise ValueError("Duplicate scenario id")
        ids.add(row["id"])
        if row["split"] not in ("train","validation","test"):
            raise ValueError("Unknown split")
        for name in ("time","latitude","longitude"):
            value=np.asarray(row["support"][name])
            if value.shape!=(2,) or not np.isfinite(value).all() or value[0]>=value[1]:
                raise ValueError("Invalid interpolation support")
    conflicts=[]
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a["split"]==b["split"]: continue
            temporal=intervals_overlap(a["support"]["time"],b["support"]["time"])
            spatial=(intervals_overlap(a["support"]["latitude"],b["support"]["latitude"])
                     and intervals_overlap(a["support"]["longitude"],b["support"]["longitude"]))
            if temporal or spatial:
                conflicts.append(dict(first=a["id"],second=b["id"],temporal=temporal,spatial=spatial))
    if conflicts: raise ValueError("Cross-split environmental overlap: "+str(conflicts))
    return dict(status="disjoint",scenarios=len(rows),rule="disjoint time support AND disjoint geographic support")
