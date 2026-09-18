import argparse,csv,datetime,json
from pathlib import Path
import numpy as np
import netCDF4 as nc

def load_csv(p):
 rows=[]
 with Path(p).open() as f:
  r=csv.DictReader(f); next(r)
  for x in r:
   try: t=datetime.datetime.fromisoformat(x['time'].replace('Z','+00:00')).replace(tzinfo=None); rows.append((x['ID'],t,float(x['latitude']),float(x['longitude']),float(x['ve']),float(x['vn'])))
   except (KeyError,ValueError): continue
 return rows

def collocate(csv_path,nc_path,out,window_h=3):
 rows=load_csv(csv_path); groups={}
 for r in rows: groups.setdefault(r[0],[]).append(r)
 with nc.Dataset(nc_path) as ds:
  lat=np.asarray(ds['latitude'][:]); lon=np.asarray(ds['longitude'][:]); tv=ds['time']; tn=nc.num2date(tv[:],tv.units,calendar=getattr(tv,'calendar','standard')); tn=np.array([datetime.datetime(x.year,x.month,x.day,x.hour,x.minute,x.second) for x in tn])
  u=np.ma.filled(ds['total_u'][:],np.nan); v=np.ma.filled(ds['total_v'][:],np.nan)
  segments=[]; skipped=0
  for ident,rr in groups.items():
   rr=sorted(rr,key=lambda x:x[1])
   for j in range(len(rr)-3):
    w=rr[j:j+4]
    if (w[-1][1]-w[0][1]).total_seconds()!=window_h*3600: continue
    if not all(lat.min()<=x[2]<=lat.max() and lon.min()<=x[3]<=lon.max() for x in w): continue
    inds=[]; good=True
    for _,t,la,lo,ve,vn in w:
     ti=int(np.argmin(np.abs(np.array([(z-t).total_seconds() for z in tn])))); yi=int(np.argmin(abs(lat-la))); xi=int(np.argmin(abs(lon-lo)))
     if not (np.isfinite(u[ti,yi,xi]) and np.isfinite(v[ti,yi,xi])): good=False; break
     inds.append((t,la,lo,ve,vn,u[ti,yi,xi],v[ti,yi,xi]))
    if good: segments.append((ident,inds))
    else: skipped+=1
 out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(out,track_id=np.array([x[0] for x in segments]),times=np.array([[z[0].replace(tzinfo=datetime.timezone.utc).timestamp() for z in x[1]] for x in segments]),positions_deg=np.array([[[z[2],z[1]] for z in x[1]] for x in segments]),velocities_ms=np.array([[[z[3],z[4]] for z in x[1]] for x in segments]),environment_uv_ms=np.array([[[z[5],z[6]] for z in x[1]] for x in segments]))
 report={'csv':str(csv_path),'netcdf':str(nc_path),'segments':len(segments),'skipped_invalid':skipped,'hours_per_segment':window_h,'source_variables':['total_u','total_v'],'interpolation':'nearest-neighbor quality adapter; no false subgrid precision','training_use':'segments must be split by track_id and time before learning'}
 (out.with_suffix('.json')).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--netcdf',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); collocate(a.csv,a.netcdf,a.out)
