"""Finite-buffer value sampling with corrections tied to actual draw probabilities."""
import numpy as np


class ValueReplaySampler:
    def __init__(self, priority):
        w=np.asarray(priority,dtype=np.float64)
        if w.ndim!=1 or len(w)==0 or not np.isfinite(w).all() or (w<=0).any():
            raise ValueError("priorities must be finite and strictly positive")
        # Scale before summing to avoid overflow; probabilities define the contract.
        w=w/w.max()
        self.probability=w/w.sum()
        if (self.probability<=0).any():
            raise ValueError("priority dynamic range underflows sampling probability")
        self.correction=1/(len(w)*self.probability)

    def sample(self,rng,size):
        if size<1: raise ValueError("positive batch size required")
        index=rng.choice(len(self.probability),size=size,p=self.probability)
        return index,self.correction[index].copy()
