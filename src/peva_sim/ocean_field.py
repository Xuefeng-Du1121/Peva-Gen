"""Vectorized trilinear interpolation. UTC seconds, degrees, m/s; no extrapolation."""
import calendar
import numpy as np
import netCDF4 as nc

class OceanField:
    def __init__(self, path, u="total_u", v="total_v"):
        with nc.Dataset(path) as ds:
            self.lat = np.asarray(ds["latitude"][:], dtype=float)
            self.lon = np.asarray(ds["longitude"][:], dtype=float)
            tv = ds["time"]
            dates = nc.num2date(tv[:], tv.units, calendar=getattr(tv, "calendar", "standard"))
            self.time = np.array([calendar.timegm((d.year,d.month,d.day,d.hour,d.minute,d.second)) for d in dates], dtype=float)
            self.u = np.ma.filled(ds[u][:], np.nan).astype(float)
            self.v = np.ma.filled(ds[v][:], np.nan).astype(float)
            for name in (u,v):
                if ds[name].dimensions != ("time","latitude","longitude"):
                    raise ValueError("Expected surface (time,latitude,longitude) variable")
                if getattr(ds[name],"units","") not in ("m/s","m s-1"):
                    raise ValueError("Velocity must be m/s")
        for axis in (self.time,self.lat,self.lon):
            if len(axis)<2 or not np.isfinite(axis).all() or not (np.diff(axis)>0).all():
                raise ValueError("Strictly increasing coordinate axes required")
        self.path = str(path)

    @staticmethod
    def _bracket(axis, q):
        if not np.isfinite(q).all() or np.any(q<axis[0]) or np.any(q>axis[-1]):
            raise ValueError("Query outside ocean field coverage")
        i = np.clip(np.searchsorted(axis,q,side="right")-1,0,len(axis)-2)
        return i, (q-axis[i])/(axis[i+1]-axis[i])

    def query(self, timestamp_s, lat, lon):
        t,y,x = np.broadcast_arrays(timestamp_s,lat,lon)
        it,ft = self._bracket(self.time,t)
        iy,fy = self._bracket(self.lat,y)
        ix,fx = self._bracket(self.lon,x)
        result = np.zeros(t.shape+(2,))
        for a in (0,1):
            for b in (0,1):
                for c in (0,1):
                    weight=(ft if a else 1-ft)*(fy if b else 1-fy)*(fx if c else 1-fx)
                    uv=np.stack((self.u[it+a,iy+b,ix+c],self.v[it+a,iy+b,ix+c]),axis=-1)
                    active=weight>1e-14
                    if np.any(active & ~np.isfinite(uv).all(axis=-1)):
                        raise ValueError("Interpolation touches missing/land cells")
                    result += np.where(active[...,None],uv,0)*weight[...,None]
        return result

    def close(self):
        """Arrays are loaded with a context manager; no open file handle remains."""
