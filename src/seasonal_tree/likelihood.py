"""Conditional multinomial tip likelihood; zeros are not floored."""
import numpy as np
from . import tree_base as base

def q_tip(history):
    n=history[1][base.TIP_EDGES]
    return np.divide(n,n.sum(1)[:,None],out=np.zeros_like(n),where=n.sum(1)[:,None]>0)

def likelihood(q,counts,epsilon=0.):
    p=q[counts>0]
    if epsilon:p=(1-epsilon)*p+epsilon/q.shape[1]
    if (p<=0).any():return -np.inf
    return float(np.dot(counts[counts>0],np.log(p)))
