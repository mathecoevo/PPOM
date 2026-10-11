"""One-region root sampler. All root proposals preserve exactly one region.

The root prior is uniform over 27 public regions, not over region subsets.
The previously verified history/i-SIR proposals remain unchanged. Whole-root
and root/history proposals are symmetric, so no root proposal correction is
needed. The ecological forward equations and effective clock are unchanged.
"""
import numpy as np
from scipy.special import logsumexp
from .events import EventTree
from .fast import base
from .likelihood import likelihood, q_tip

class SingleRootTree(EventTree):
    root_region_count=27
    def root(self,mask,fraction=1.):
        r=int(mask)
        if r<=0 or r.bit_count()!=1 or r>=1<<self.root_region_count:
            raise ValueError('Root must contain exactly one of the public regions')
        return super().root(r,fraction)
    def manifest(self):
        value=super().manifest()
        value.update(root_prior='uniform over the 27 single-region origins',
                     root_exactly_one_region=True,root_regions=27,
                     root_cell_initial_abundance='K on selected region; zero elsewhere')
        return value

def propose(model,R,u,h,rng,mode,counts,epsilon):
    assert mode in ("known", "unknown") and int(R).bit_count() == 1
    nr = model.root_region_count
    v=u.copy();extra=0.;kind=2
    if hasattr(model,'offset') and rng.random()<.12:
        # Conditional importance-resampling (i-SIR): current state plus seven
        # independent draws from the conditional branch-innovation prior.
        # This is reversible for the conditional posterior. Its kernel ratio
        # cancels the outer MH likelihood ratio, so it is accepted directly.
        leaves=np.array([0,1,3,4,5]);observed=leaves[counts.sum(1)>0]
        edge=int(rng.choice(observed));a,b=model.offset[edge:edge+2]
        states=[(R,u,h,q_tip(h),likelihood(q_tip(h),counts,epsilon),0.)]
        for _ in range(7):
            trial=u.copy();trial[a:b]=rng.integers(2**31-1,size=b-a)
            g=model.run_noise(R,trial,h,(edge,0))
            if g[3]:states.append(None);continue
            p=q_tip(g);l=likelihood(p,counts,epsilon);states.append((R,trial,g,p,l,0.))
        score=np.array([s[4] if s is not None else -np.inf for s in states])
        if not np.isfinite(score).any():return None,kind
        weights=np.exp(score-logsumexp(score));j=int(rng.choice(len(states),p=weights))
        r,v,g,p,l,_=states[j]
        correction=float(score[0]-l) if np.isfinite(score[0]) else 0.
        return (r,v,g,p,l,correction),kind
    choice=rng.random()
    if mode=='unknown' and choice<.08:
        kind=0
        trial=1 << int(rng.integers(nr))
        v=rng.integers(2**31-1,size=len(u))
        candidate=model.run_noise(trial,v)
    elif mode=='unknown' and choice<.35:
        kind=1
        index=int(R).bit_length()-1
        new_index=int(rng.integers(nr-1))
        if new_index>=index:new_index+=1
        trial=1 << new_index
        candidate=model.run_noise(trial,v)
    else:
        trial=R;changed=int(rng.integers(12))
        if hasattr(model,'offset'):
            if changed<8:
                events=int(model.data['times'][changed])
                # Mixture has full support at every interval. Early and late
                # edits both have state-independent symmetric proposal laws.
                early=rng.random()<.5
                event=int(rng.integers(events)) if early else int(rng.integers(max(0,events-40),events))
                width=int(rng.choice([1,5,20,100],p=[.4,.3,.2,.1]))
                ix=np.arange(model.offset[changed]+event,min(model.offset[changed]+event+width,model.offset[changed+1]))
            else:
                event=0;ix=np.array([model.offset[-1]+changed-8])
            v[ix]=rng.integers(2**31-1,size=len(ix))
            candidate=model.run_noise(trial,v,h,(changed,event))
        elif changed in (2,6,7) and rng.random()<.15:
            block={2:[2,0,1,10],6:[6,2,0,1,3,9,10],7:[7,4,5,11]}[changed]
            v[block]=rng.integers(2**31-1,size=len(block))
            candidate=model.run_noise(trial,v,h[:3],changed)
        else:
            v[changed]=rng.integers(2**31-1)
            candidate=model.run_noise(trial,v,h[:3],changed)
    assert int(trial).bit_count() == 1
    if candidate[3]:return None,kind
    q=q_tip(candidate);ll=likelihood(q,counts,epsilon)
    return (trial,v,candidate,q,ll,extra),kind
