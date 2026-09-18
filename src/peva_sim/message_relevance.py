"""Message relevance from only the received public PVF grid.

Analytic derivative of a bilinear grid interpolant, not the exact particle KDE
derivative used centrally during training. No target truth/critic state is read.
"""
import numpy as np


def public_grid_gradient(observation,xy):
    grid=np.asarray(observation["prior_grid_m"],dtype=float)
    value=np.asarray(observation["value_density"],dtype=float)
    xy=np.asarray(xy,dtype=float)
    if grid.ndim!=2 or grid.shape[1]!=2 or value.shape!=(len(grid),) or xy.ndim!=2 or xy.shape[1]!=2:
        raise ValueError("invalid public grid/query shape")
    if not all(np.isfinite(a).all() for a in (grid,value,xy)):
        raise ValueError("nonfinite public grid/query")
    xs=np.unique(grid[:,0]);ys=np.unique(grid[:,1])
    if min(len(xs),len(ys))<2 or len(xs)*len(ys)!=len(grid):
        raise ValueError("complete rectangular grid required")
    table=np.full((len(ys),len(xs)),np.nan)
    table[np.searchsorted(ys,grid[:,1]),np.searchsorted(xs,grid[:,0])]=value
    if not np.isfinite(table).all(): raise ValueError("duplicate/missing grid cells")
    x=np.clip(xy[:,0],xs[0],xs[-1]);y=np.clip(xy[:,1],ys[0],ys[-1])
    ix=np.clip(np.searchsorted(xs,x,side="right")-1,0,len(xs)-2)
    iy=np.clip(np.searchsorted(ys,y,side="right")-1,0,len(ys)-2)
    dx=xs[ix+1]-xs[ix];dy=ys[iy+1]-ys[iy]
    tx=(x-xs[ix])/dx;ty=(y-ys[iy])/dy
    f00=table[iy,ix];f10=table[iy,ix+1];f01=table[iy+1,ix];f11=table[iy+1,ix+1]
    gx=((1-ty)*(f10-f00)+ty*(f11-f01))/dx
    gy=((1-tx)*(f01-f00)+tx*(f11-f10))/dy
    # Constant boundary extension: zero derivative along an out-of-grid axis.
    gx[(xy[:,0]<xs[0])|(xy[:,0]>xs[-1])]=0
    gy[(xy[:,1]<ys[0])|(xy[:,1]>ys[-1])]=0
    return np.column_stack((gx,gy))


def physics_message_scores(observation,estimated_latent,config,kappa=1.):
    if not np.isfinite(kappa) or not 0<=kappa<=1:
        raise ValueError("prototype kappa must be within [0,1]")
    xy=(np.asarray(estimated_latent)+1)*config.region_m/2
    grad=public_grid_gradient(observation,xy)
    reference=1/(2*np.pi*config.prior_sigma_m**2)
    omega=np.linalg.norm(grad*(config.region_m/2)/reference,axis=-1).clip(.1,10)
    return kappa*omega
