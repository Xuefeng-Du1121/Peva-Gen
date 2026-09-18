"""Observable-only collection behaviors, not claimed as optimized SAR baselines."""
import numpy as np


class CollectionPolicy:
    MODES=("pvf","random_waypoint","lane_sweep")
    def __init__(self,config,mode,seed):
        if mode not in self.MODES: raise ValueError("unknown collection behavior")
        self.c=config;self.mode=mode;self.rng=np.random.default_rng(seed)
        self.destinations=None
        self.direction=np.ones(config.n_uavs)

    def action(self,observation):
        c=self.c;positions=np.asarray(observation["positions_m"],dtype=float)
        if self.mode=="pvf":
            p=np.asarray(observation["value_density"],dtype=float)+1e-30
            destination=observation["prior_grid_m"][self.rng.choice(len(p),size=c.n_uavs,p=p/p.sum())]
            delta=destination-positions
            velocity=c.speed_mps*(.8*delta/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-12)
                                  +.2*self.rng.uniform(-1,1,(c.n_uavs,2)))
        else:
            if self.destinations is None:
                if self.mode=="random_waypoint":
                    self.destinations=self.rng.uniform(0,c.region_m,(c.n_uavs,2))
                else:
                    self.destinations=np.column_stack(((np.arange(c.n_uavs)+.5)*c.region_m/c.n_uavs,
                                                        np.full(c.n_uavs,.95*c.region_m)))
            arrived=np.linalg.norm(self.destinations-positions,axis=-1)<c.speed_mps*c.dt_s
            if self.mode=="random_waypoint":
                self.destinations[arrived]=self.rng.uniform(0,c.region_m,(arrived.sum(),2))
            else:
                self.direction[arrived]*=-1
                self.destinations[:,1]=np.where(self.direction>0,.95,.05)*c.region_m
            delta=self.destinations-positions
            velocity=c.speed_mps*delta/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-12)
        norm=np.linalg.norm(velocity,axis=-1,keepdims=True)
        return velocity*np.minimum(1,c.speed_mps/np.maximum(norm,1e-12))
