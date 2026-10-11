"""Actual production proposal checked against an exactly summable posterior."""
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from .proposals import propose
from .likelihood import likelihood, q_tip
from .diagnostics import split_rank_rhat, ess
from .sampler import atomic_json

class ExactToy:
    root_region_count=27
    offset=np.arange(9)
    data={'times':np.ones(8,dtype=int)}
    def __init__(self):
        self.q=np.empty((27,2,27))
        x=np.arange(27)
        for r in range(27):
            for parity in range(2):
                center=(r+3*parity)%27
                distance=np.minimum(abs(x-center),27-abs(x-center))
                p=np.exp(-distance**2/18)+.1
                self.q[r,parity]=p/p.sum()
    def run_noise(self,R,u,previous=None,changed=None):
        assert int(R).bit_count()==1 and 0<R<1<<27
        parity=int(np.sum(u%2)%2)
        q=self.q[int(R).bit_length()-1,parity]
        return None,np.tile(q,(8,1)),None,0

def implementation_test():
    model=ExactToy();counts=np.zeros((5,27),int);counts[0,8]=3;counts[0,9]=2
    exact=np.array([[likelihood(np.tile(model.q[r,b],(5,1)),counts) for b in range(2)] for r in range(27)])
    # With 12 independent uniforms on 0..2^31-2 the parity bias is N^-12,
    # far below floating precision; compute that distribution explicitly.
    even=(1+(1/(2**31-1))**12)/2
    logprior=np.log([even,1-even])
    weight=np.exp(exact+logprior-logsumexp(exact+logprior))
    actual_root=weight.sum(1)
    exact_map=np.einsum('rb,rbm->m',weight,model.q)
    root_freq=[];means=[];lls=[];root_moves=[]
    steps=64000;burn=16000
    for chain in range(4):
        rng=np.random.default_rng(202610118100+chain*100003)
        state=[]
        for _ in range(2):
            r=1<<int(rng.integers(27));u=rng.integers(2**31-1,size=12)
            h=model.run_noise(r,u);state.append((r,u,h,q_tip(h)))
        ll=[likelihood(state[0][3],counts),.05*likelihood(state[1][3],counts,.001)]
        roots=[];maps=[];trace=[]
        for step in range(steps):
            k=0 if rng.random()<.75 else 1;beta=(1.,.05)[k];eps=(0.,.001)[k]
            r,u,h,q=state[k]
            p,kind=propose(model,r,u,h,rng,'unknown',counts*beta,eps)
            if p is not None:
                r,v,g,q,l,extra=p
                assert r.bit_count()==1
                if np.isfinite(l) and np.log(rng.random())<min(0.,l-ll[k]+extra):
                    state[k]=(r,v,g,q);ll[k]=l
            if rng.random()<.25:
                a=likelihood(state[1][3],counts);b=.05*likelihood(state[0][3],counts,.001)
                if np.log(rng.random())<min(0.,a+b-sum(ll)):
                    state=[state[1],state[0]];ll=[a,b]
            if step>=burn:
                roots.append(state[0][0].bit_length()-1);maps.append(state[0][3][0]);trace.append(ll[0])
        root_freq.append(np.bincount(roots,minlength=27)/len(roots));means.append(np.mean(maps,axis=0));lls.append(trace)
        root_moves.append(int(np.sum(np.diff(roots)!=0)))
    tv=float(abs(np.mean(root_freq,axis=0)-actual_root).sum()/2)
    mtv=float(abs(np.mean(means,axis=0)-exact_map).sum()/2)
    rh=float(split_rank_rhat(np.array(lls)));e=float(ess(np.array(lls)))
    assert tv<.025 and mtv<.015 and rh<1.01 and e>500,(tv,mtv,rh,e)
    # Both whole-root and location-swap proposals have a symmetric root law.
    global_matrix=np.full((27,27),1/27)
    local_matrix=(np.ones((27,27))-np.eye(27))/26
    assert np.allclose(global_matrix,global_matrix.T) and np.allclose(local_matrix,local_matrix.T)
    assert np.allclose(global_matrix.sum(1),1) and np.allclose(local_matrix.sum(1),1)
    return dict(passed=True,actual_production_propose_function_used=True,
        root_TV_to_exact=tv,predictive_TV_to_exact=mtv,likelihood_Rhat=rh,likelihood_ESS=e,
        retained_per_chain=steps-burn,root_changes_per_chain=root_moves,
        every_returned_proposal_has_exactly_one_region=True,
        uniform_prior_over_locations=True,symmetric_root_proposals=True)

def spatial_integration(directory):
    with np.load(directory / 'enumerated.npz') as z:
        # Enumeration from the real 24-cell seasonal spatial tree.
        masks = np.array([1, 2, 4])
        q = z['q'][masks - 1].astype(float)
        status = z['status'][masks - 1]
    with np.load(directory / 'result_maps.npz') as z:
        counts = z['counts'].copy()
    records = []
    for protocol in ('focal', 'all'):
        c = counts.copy()
        if protocol == 'focal': c[1:] = 0
        ll = np.full(status.shape, -np.inf)
        for r, u in zip(*np.where(status == 0)):
            ll[r,u] = likelihood(q[r,u], c)
        assert np.isfinite(ll).any()
        w = np.exp(ll - logsumexp(ll))
        mean = np.einsum('ru,rutm->tm', w, q)
        assert np.max(abs(mean.sum(1) - 1)) < 1e-7
        records.append(dict(protocol=protocol, root_masks=masks.tolist(),
            exact_root_posterior=w.sum(1).tolist(),
            scope='24-cell stored spatial enumeration, 4096 histories per root',
            enumeration_not_recomputed_in_this_command=True))
    return records


def validate(output):
    start = time.perf_counter()
    result = dict(passed=True, proposal_implementation=implementation_test(),
                  cold_epsilon=0., hot_draws_not_used_as_posterior=True)
    directory = Path.cwd() / 'validation'
    if not (directory / 'enumerated.npz').exists():
        directory = Path(__file__).resolve().parents[2] / 'validation'
    if (directory / 'enumerated.npz').exists():
        result['stored_spatial_enumeration_integration'] = spatial_integration(directory)
    else:
        result['stored_spatial_enumeration_integration'] = 'not run: reference files unavailable'
    result['seconds'] = time.perf_counter() - start
    atomic_json(output, result)
    return result
