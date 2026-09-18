"""Gaussian-reference parameterization of the 128-D transport belief."""
from .latent_transport import LatentTransportJointBelief
from .gaussian_reference import GaussianReferenceVNet

class ReferenceTransportJointBelief(LatentTransportJointBelief):
    architecture="transport-latent128-gaussian-reference-v2"
    def __init__(self,context_dim,codec,mean=None,covariance=None):
        super().__init__(context_dim,codec)
        net=GaussianReferenceVNet(128,context_dim+32,mean,covariance,self.belief.training_steps)
        net.alpha_bar.copy_(self.belief.alpha_bar)
        self.belief.eps=net
