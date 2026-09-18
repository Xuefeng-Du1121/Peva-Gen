"""Confirmed target ID payload, separate from CIB and shared grid.

Payload is uint32 count followed by sorted unique uint32 IDs (little endian).
No hidden coordinates are transmitted. A mission boundary requires reset.
Link headers, uplink detection reports, delay/loss and grid bytes are accounted
by the transport caller, not silently treated as free here.
"""
import struct
import numpy as np

def encode_confirmations(ids,n_targets):
    values=list(ids)
    if not isinstance(n_targets,int) or not 1<=n_targets<2**32:
        raise ValueError("Invalid target count")
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer))
           or not 0<=x<n_targets for x in values):
        raise ValueError("Invalid confirmed target ID")
    if len(set(values))!=len(values):
        raise ValueError("Duplicate target ID")
    values=sorted(values)
    return struct.pack("<I",len(values))+b"".join(struct.pack("<I",x) for x in values)

def decode_confirmations(payload,n_targets):
    if not isinstance(payload,bytes) or len(payload)<4:
        raise ValueError("Invalid confirmation payload")
    count=struct.unpack_from("<I",payload)[0]
    if count>n_targets or len(payload)!=4+4*count:
        raise ValueError("Invalid confirmation length")
    ids=[struct.unpack_from("<I",payload,4+4*i)[0] for i in range(count)]
    if encode_confirmations(ids,n_targets)!=payload:
        raise ValueError("Noncanonical confirmation payload")
    return ids

class ConfirmationLedger:
    def __init__(self,n_targets):
        encode_confirmations([],n_targets)
        self.n_targets=n_targets
        self.reset()
    def reset(self):
        self._confirmed=np.zeros(self.n_targets,dtype=bool)
        self.received_payload_bytes=0
    def receive(self,payload):
        ids=decode_confirmations(payload,self.n_targets)
        self._confirmed[ids]=True
        self.received_payload_bytes+=len(payload)
    @property
    def confirmed(self):
        return self._confirmed.copy()
