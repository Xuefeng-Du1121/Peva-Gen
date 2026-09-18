"""Strict adapter for recorded drift tracks.
CSV schema: track_id,timestamp,x_m,y_m,current_u_mps,current_v_mps,wind_u_mps,wind_v_mps.
Coordinates must be local projected metres; timestamps ISO-8601 or seconds.
"""
from pathlib import Path
import csv,datetime
import numpy as np
REQUIRED=('track_id','timestamp','x_m','y_m','current_u_mps','current_v_mps','wind_u_mps','wind_v_mps')
def load_tracks(path):
 path=Path(path)
 if path.suffix.lower()!='.csv': raise ValueError('only explicit CSV adapter is supported')
 rows=[]
 with path.open(newline='') as f:
  reader=csv.DictReader(f)
  if tuple(reader.fieldnames or ()) != REQUIRED: raise ValueError(f'CSV columns must equal {REQUIRED}')
  for r in reader:
   try:
    ts=float(r['timestamp'])
   except ValueError: ts=datetime.datetime.fromisoformat(r['timestamp'].replace('Z','+00:00')).timestamp()
   vals=[float(r[k]) for k in REQUIRED[2:]]
   rows.append((r['track_id'],ts,*vals))
 if not rows: raise ValueError('empty track file')
 arr=np.asarray([x[1:] for x in rows],dtype=float)
 if not np.isfinite(arr).all(): raise ValueError('non-finite drift value')
 return {'track_id':np.asarray([x[0] for x in rows]),'timestamp_s':arr[:,0],'x_m':arr[:,1],'y_m':arr[:,2],'current_uv':arr[:,3:5],'wind_uv':arr[:,5:7]}
