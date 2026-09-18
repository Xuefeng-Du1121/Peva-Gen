"""Opt-in reliable shared-grid confirmation broadcast, protocol v1.

Confirmed IDs are sensor outcomes, not true target coordinates. Simulator-side
access to found is confined to the broadcaster after the world/sensor step.
Snapshot payload is piggybacked on the existing shared-grid packet: no second
header. Extra payload bytes are added to shared_downlink. The existing reliable,
zero-delay shared-grid assumption is explicit; peer packet_drop does not apply.
Old environments and training jobs are unchanged.
"""
import numpy as np
from .confirmation_protocol import ConfirmationLedger,encode_confirmations

class ConfirmedEnvironment:
    protocol_version="shared-confirmation-snapshot-v1"
    def __init__(self,env):
        self.env=env
        self.ledger=ConfirmationLedger(env.cfg.n_targets)
    def __getattr__(self,name):
        return getattr(self.env,name)
    def _observation(self,obs):
        return {**obs,"confirmed_targets":self.ledger.confirmed}
    def _info(self,info):
        return {**info,"bytes":{**info["bytes"],
                "shared_downlink":info["bytes"]["shared_downlink"]+
                                  self.ledger.received_payload_bytes}}
    def reset(self,seed=0):
        obs,info=self.env.reset(seed)
        self.ledger.reset()
        return self._observation(obs),self._info(info)
    def reset_for_training_scenario(self,seed,scenario_id):
        obs,info=self.env.reset_for_training_scenario(seed,scenario_id)
        self.ledger.reset()
        return self._observation(obs),self._info(info)
    def step(self,*args,**kwargs):
        obs,reward,terminated,truncated,info=self.env.step(*args,**kwargs)
        packet=encode_confirmations(np.flatnonzero(self.env.found),self.cfg.n_targets)
        self.ledger.receive(packet)
        return self._observation(obs),reward,terminated,truncated,self._info(info)
    def observation(self):
        return self._observation(self.env.observation())
    def local_observations(self):
        return {i:self._observation(obs) for i,obs in self.env.local_observations().items()}
    def info(self):
        return self._info(self.env.info())
