"""Simulated target drift driven by gridded currents, never recorded target tracks.

Historical analysis forcing is NOT an issue-time forecast. No wind/leeway model
is claimed without a matching wind product. Simulation uncertainty is explicit:
per-episode true diffusivity and a bounded unresolved drift vector; the particle
belief retains nominal diffusivity and has no access to the true drift vector.
"""
import numpy as np
from .env import MaritimeSAR,Config,reflect
from .real_env import LocalFrame

class OceanForcedSAR(MaritimeSAR):
    def __init__(self,field,origin_lonlat,start_utc,config,truth_diffusion_range=(.5,4.),drift_error_bound_mps=.1):
        if config.windage!=0 or config.forecast_bias_mps!=0:
            raise ValueError("Use zero windage/scalar bias; wind unavailable and vector error modeled separately")
        origin=np.asarray(origin_lonlat,dtype=float)
        limits=np.asarray(truth_diffusion_range,dtype=float)
        if origin.shape!=(2,) or not np.isfinite(origin).all() or not np.isfinite(start_utc):
            raise ValueError("Finite origin and UTC time required")
        if limits.shape!=(2,) or not np.isfinite(limits).all() or not 0<=limits[0]<=limits[1]:
            raise ValueError("Invalid truth diffusion range")
        if not np.isfinite(drift_error_bound_mps) or drift_error_bound_mps<0:
            raise ValueError("Invalid drift error bound")
        super().__init__(config)
        self.field=field;self.start_utc=float(start_utc)
        self.frame=LocalFrame(origin,config.region_m)
        self.truth_diffusion_range=tuple(limits)
        self.drift_error_bound_mps=drift_error_bound_mps
        corners=np.array([[0,0],[0,config.region_m],[config.region_m,0],[config.region_m,config.region_m]])
        ll=self.frame.inverse(corners)
        for t in (self.start_utc,self.start_utc+config.horizon_s):
            field.query(t,ll[:,1],ll[:,0])
    def reset(self,seed=0):
        parameter_rng=np.random.default_rng(np.random.SeedSequence([seed,812713]))
        self.truth_diffusion_m2s=float(parameter_rng.uniform(*self.truth_diffusion_range))
        angle=parameter_rng.uniform(0,2*np.pi)
        magnitude=self.drift_error_bound_mps*np.sqrt(parameter_rng.uniform())
        self.truth_drift_error_mps=magnitude*np.array([np.cos(angle),np.sin(angle)])
        return super().reset(seed)
    def flow(self,xy,t):
        ll=self.frame.inverse(xy)
        return self.field.query(self.start_utc+t,ll[...,1],ll[...,0])
    def advance_targets(self,next_time):
        c=self.cfg
        if not np.isclose(next_time,self.t+c.dt_s):
            raise ValueError("One fixed integration step required")
        velocity=self.flow(self.targets,self.t)+self.truth_drift_error_mps
        noise=self.world_rng.normal(0,np.sqrt(2*self.truth_diffusion_m2s*c.dt_s),self.targets.shape)
        return reflect(self.targets+velocity*c.dt_s+noise,c.region_m)
    def forcing_metadata(self):
        return dict(protocol="simulated-targets/historical-current-analysis-v1",
                    origin_lonlat=self.frame.origin.tolist(),start_utc=self.start_utc,
                    wind_product=None,windage=0.,truth_diffusion_range=list(self.truth_diffusion_range),
                    drift_error_bound_mps=self.drift_error_bound_mps,
                    truth_tracks="generated online; no recorded trajectory used",
                    limitation="not an issue-time forecast; no geographic/date holdout guaranteed by this environment alone")
    def close(self):
        self.field.close()
