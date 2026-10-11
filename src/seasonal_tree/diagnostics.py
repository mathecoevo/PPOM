"""Rank-normalized split Rhat and autocorrelation ESS helpers."""
import numpy as np
from scipy.stats import rankdata,norm

def finite_number(x):
    return float(x) if np.isfinite(x) else None

def split_rank_rhat(x):
    x=np.asarray(x,dtype=float);M,N=x.shape
    N=N//2
    if M<2 or N<4:return np.nan
    y=np.concatenate((x[:,:N],x[:,-N:]),axis=0)
    if np.ptp(y)<1e-12:return 1.
    z=norm.ppf((rankdata(y.ravel()).reshape(y.shape)-.375)/(y.size+.25))
    def rhat(a):
        W=np.mean(np.var(a,axis=1,ddof=1));B=N*np.var(a.mean(1),ddof=1)
        if W==0:return 1. if B==0 else np.inf
        return float(np.sqrt(((N-1)*W/N+B/N)/W))
    folded=norm.ppf((rankdata(abs(y-np.median(y)).ravel()).reshape(y.shape)-.375)/(y.size+.25))
    return max(rhat(z),rhat(folded))

def ess(x):
    x=np.asarray(x,dtype=float);M,N=x.shape
    if np.ptp(x)<1e-12:return float(M*N)
    a=x-x.mean(1)[:,None]
    f=np.fft.rfft(a,n=2*N,axis=1);cov=np.fft.irfft(f*np.conj(f),axis=1)[:,:N]/N
    W=cov[:,0].mean();B=N*np.var(x.mean(1),ddof=1)
    v=(N-1)*W/N+B/N
    if v<=0:return float(M*N)
    rho=1-(W-cov.mean(0))/v;total=0.;previous=np.inf
    for k in range(1,N-1,2):
        pair=rho[k]+rho[k+1]
        if pair<0:break
        pair=min(pair,previous);total+=pair;previous=pair
    return float(min(M*N,M*N/(1+2*total)))

