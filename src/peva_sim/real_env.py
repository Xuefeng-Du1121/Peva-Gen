"""Recorded-drifter replay with independent gridded ocean forcing for beliefs.

This is a one-drifter SAR benchmark, not three replicated survivors.
Ocean fields are retrospective analysis data, not issue-time forecasts.
"""
from pathlib import Path
from dataclasses import replace
import numpy as np
from .env import MaritimeSAR, Config
from .real_replay import RealReplay
from .ocean_field import OceanField

class LocalFrame:
    """Local spherical equirectangular frame (metres); suitable for a 40 km patch."""
    radius = 6371008.8
    def __init__(self, lonlat0, region_m):
        self.origin = np.asarray(lonlat0,dtype=float)
        self.center = region_m/2
        self.scales = self.radius*np.pi/180*np.array([np.cos(np.deg2rad(self.origin[1])),1.])
    def project(self, lonlat):
        p=np.asarray(lonlat,dtype=float)-self.origin
        return p*self.scales+self.center
    def inverse(self, xy):
        return (np.asarray(xy)-self.center)/self.scales+self.origin

class RealMaritimeSAR(MaritimeSAR):
    def __init__(self, track_file, config=None, split="test", seed=0, field_path=None, field=None, manifest_path=None):
        config = config or replace(Config(),n_targets=1)
        if config.n_targets != 1:
            raise ValueError("One recorded track per episode; do not duplicate it into multiple targets")
        if config.horizon_s > 10800:
            raise ValueError("Recorded segments cover at most three hours")
        super().__init__(config)
        self.replay = RealReplay(track_file,split,seed,manifest_path=manifest_path)
        if field is None:
            if field_path is None:
                root=Path(__file__).resolve().parents[2]
                field_path=root/"ocean_system_data_20260913T024902Z/drift-trajectory-system/data_platform/data/raw/environment/copernicus/multobs_currents_east_china_2018_01.nc"
            field=OceanField(field_path)
        self.field=field

    def reset(self, seed=0, episode_index=None):
        index = seed % len(self.replay.indices) if episode_index is None else episode_index
        self.episode=self.replay.reset(index)
        self.absolute_times=np.asarray(self.episode["times"],dtype=float)
        if not np.isfinite(self.absolute_times).all() or not np.all(np.diff(self.absolute_times)==3600):
            raise ValueError("Require four contiguous hourly UTC samples")
        self.start_utc=float(self.absolute_times[0])
        self.frame=LocalFrame(self.episode["positions_deg"][0],self.cfg.region_m)
        self.path_m=self.frame.project(self.episode["positions_deg"])
        if np.any(self.path_m<0) or np.any(self.path_m>self.cfg.region_m):
            raise ValueError("Track leaves local patch; no coordinate rescaling allowed")
        super().reset(seed)
        self.targets=self.path_m[0:1].copy()
        # Public last-known datum is the first record; future track is privileged.
        # Particles remain independent draws around that datum from parent reset.
        return self.observation(), self.info()

    def advance_targets(self, next_time):
        if next_time<0 or next_time>self.absolute_times[-1]-self.start_utc:
            raise ValueError("Recorded trajectory horizon exceeded")
        offsets=self.absolute_times-self.start_utc
        return np.array([[np.interp(next_time,offsets,self.path_m[:,j]) for j in range(2)]])

    def flow(self, xy, t):
        lonlat=self.frame.inverse(xy)
        # Query every particle's position; never use currents collocated at future true positions.
        return self.field.query(self.start_utc+t,lonlat[...,1],lonlat[...,0])

    def close(self):
        self.field.close()
