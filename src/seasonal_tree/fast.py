"""Numerical acceleration of the certified mentor adapter, without model changes.

The fixed-two offspring survival equation is solved by monotone nonlinear
Gauss-Seidel, eliminating its diagonal quadratic exactly. Spectral tests are
still used when row bounds and the diagonal do not certify criticality.
"""
from pathlib import Path
import time
from . import mentor as base
from .mentor import np, sp, njit, eigsh


@njit(cache=True)
def component_newton(sf,component_cells,qp,qi,qw):
    N=len(component_cells);local=np.full(len(sf),-1,dtype=np.int64)
    for k in range(N):local[component_cells[k]]=k
    A=np.zeros((N,N))
    for c in range(N):
        g=component_cells[c]
        for j in range(qp[g],qp[g+1]):
            d=qi[j];A[c,local[d]]=qw[j]*sf[d]
    pi=np.ones(N)
    for _ in range(6):
        s=A@pi;pi=s*(2-s)
    for iteration in range(100):
        s=A@pi;residual=pi-s*(2-s);error=np.max(np.abs(residual))
        if error<1e-11:return pi
        J=np.eye(N)
        for c in range(N):
            for d in range(N):J[c,d]-=2*(1-s[c])*A[c,d]
        correction=np.linalg.solve(J,residual);scale=1.;accepted=False
        for _ in range(60):
            trial=pi-scale*correction
            if np.min(trial)>=0 and np.max(trial)<=1:
                st=A@trial;r=trial-st*(2-st)
                if np.max(np.abs(r))<error:accepted=True;break
            scale*=.5
        if not accepted:raise RuntimeError('Survival Newton fallback line search failed')
        pi=trial
    raise RuntimeError('Survival Newton fallback did not converge')


@njit(cache=True)
def needed_survival(sf,new,present,K,cp,ci,qp,qi,qw,warm=None):
    """Solve only components whose absent cells have nonzero arrivals.

    A vacant cell has sf>0.5/0.98 on this map (verified at each solve),
    certifying supercriticality of every component evaluated here.
    """
    pi=np.ones(len(K)) if warm is None else warm.copy()
    for component in range(len(cp)-1):
        needed=False; certified=False
        for k in range(cp[component],cp[component+1]):
            c=ci[k]
            if not present[c] and new[c]>0:
                needed=True
                for j in range(qp[c],qp[c+1]):
                    if qi[j]==c and 2*qw[j]*sf[c]>1+1e-11: certified=True
        if not needed: continue
        if not certified: raise RuntimeError('Vacancy did not certify supercriticality; use reference solver')
        converged=False
        for iteration in range(200):
            change=0.
            for k in range(cp[component],cp[component+1]):
                c=ci[k];a=0.;b=0.
                for j in range(qp[c],qp[c+1]):
                    d=qi[j];w=qw[j]*sf[d]
                    if d==c:a=w
                    else:b+=w*pi[d]
                B=1-2*a+2*a*b;C=b*b-2*b;disc=np.sqrt(max(0.,B*B-4*a*a*C))
                if a==0:value=-C
                elif B>=0:value=0. if disc+B==0 else -2*C/(disc+B)
                else:value=(-B+disc)/(2*a*a)
                value=min(1.,max(0.,value));change=max(change,abs(value-pi[c]));pi[c]=value
            # A quadratic cancellation can make successive iterates fluctuate
            # above machine-level change tolerances even after the equation is
            # solved. Certify its residual rather than iteration differences.
            if change<2e-13 or iteration%25==24:
                residual=0.
                for k in range(cp[component],cp[component+1]):
                    c=ci[k];s=0.
                    for j in range(qp[c],qp[c+1]):s+=qw[j]*sf[qi[j]]*pi[qi[j]]
                    residual=max(residual,abs(pi[c]-s*(2-s)))
                if residual<1e-11:converged=True;break
        if not converged:
            cells=ci[cp[component]:cp[component+1]]
            solved=component_newton(sf,cells,qp,qi,qw)
            for k in range(len(cells)):pi[cells[k]]=solved[k]
        for k in range(cp[component],cp[component+1]):
            c=ci[k];s=0.
            for j in range(qp[c],qp[c+1]):s+=qw[j]*sf[qi[j]]*pi[qi[j]]
            if abs(pi[c]-s*(2-s))>1e-11:raise RuntimeError('Uncertified needed survival fixed point')
    return pi


@njit(cache=True)
def branch_jit(n0,events,seed,K,wp,wi,ww,gamma,cp,ci,qp,qi,qw,jump=True,max_updates=20000):
    np.random.seed(seed);n=n0.copy();ever=n>0;present=n>0
    if events>len(n)-ever.sum():return n,np.array([0,0,0,0,1],dtype=np.int64)
    ticks=0;updates=0;skipped=0;jumps=0;stable=0;warm=np.ones(len(n))
    while ticks<events:
        if n.sum()==0 or ever.all():return n,np.array([ticks,updates,skipped,jumps,1],dtype=np.int64)
        if updates>=max_updates:raise RuntimeError('Branch update limit, not a zero likelihood')
        new=base_collapsed(n/K,K,wp,wi,gamma)
        sf=site_factor(n,K,wp,wi,ww)
        pi=needed_survival(sf,new,present,K,cp,ci,qp,qi,qw,warm);warm=pi
        lam=np.zeros(len(n));predicted=np.zeros(len(n));stationary=True
        for d in range(len(n)):
            if present[d]:
                predicted[d]=K[d]*new[d]
                if abs(predicted[d]-n[d])>=1e-9 or 0<predicted[d]<1:stationary=False
            else:lam[d]=K[d]*new[d]*pi[d]
        hit=np.zeros(len(n),dtype=np.bool_)
        if jump and stable>=10 and stationary:
            total=lam.sum()
            if total<=0:return n,np.array([ticks,updates,skipped,jumps,1],dtype=np.int64)
            waiting=np.random.geometric(-np.expm1(-total));skipped+=waiting;jumps+=1
            count=positive_poisson_jit(total)
            for _ in range(count):
                u=np.random.random()*total;cumulative=0.
                for d in range(len(n)):
                    cumulative+=lam[d]
                    if u<cumulative:hit[d]=True;break
        else:
            for d in range(len(n)):
                if lam[d]>0:hit[d]=np.random.random() < -np.expm1(-lam[d])
            updates+=1
        gained=False
        for d in range(len(n)):
            if hit[d]:predicted[d]=min(1/max(pi[d],1e-3),K[d]);present[d]=True
            if predicted[d]<1:predicted[d]=0.;present[d]=False
            if predicted[d]>0 and not ever[d]:ever[d]=True;gained=True
        if gained:ticks+=1;stable=0
        else:stable=stable+1 if stationary else 0
        n=predicted
    return n,np.array([ticks,updates,skipped,jumps,0],dtype=np.int64)


base_collapsed=base.collapsed_map
positive_poisson_jit=base._old.positive_poisson


class JitTree(base.MentorTree):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        m=self.landscape
        self.cp=np.r_[0,np.cumsum([len(x) for x in m.components])].astype(np.int64)
        self.ci=np.concatenate(m.components).astype(np.int64)
    def run_branch(self,n,events,seed,jump=None,max_updates=20000):
        m=self.landscape;q=m.clonal;w=m.W
        return branch_jit(np.asarray(n,dtype=float),int(events),int(seed),m.K,
            w.indptr,w.indices,w.data,m.gamma,self.cp,self.ci,q.indptr,q.indices,q.data,
            self.jump if jump is None else jump,max_updates)
    def manifest(self):
        result=super().manifest()
        import hashlib
        result.update(numerical_kernel='Numba hard-ecotype specialization of supplied regional model',
            pi_solver='certified vacant component + monotone Gauss-Seidel with exact diagonal quadratic',
            sha256_numerical_kernel=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            seed_correspondence='Numba MT random streams; distribution preserved, individual histories differ from PCG streams')
        return result


@njit(cache=True)
def survival_gs(indptr, indices, weights, factor, tolerance=2e-13):
    pi=np.ones(len(factor))
    for iteration in range(20000):
        change=0.
        for c in range(len(pi)):
            a=0.; b=0.
            for j in range(indptr[c],indptr[c+1]):
                d=indices[j]; w=weights[j]*factor[d]
                if d==c: a=w
                else: b+=w*pi[d]
            B=1-2*a+2*a*b; C=b*b-2*b
            disc=np.sqrt(max(0.,B*B-4*a*a*C))
            if a==0: value=-C
            elif B>=0:
                value=0. if disc+B==0 else -2*C/(disc+B)
            else: value=(-B+disc)/(2*a*a)
            value=min(1.,max(0.,value))
            change=max(change,abs(value-pi[c])); pi[c]=value
        if change<tolerance:
            residual=0.
            for c in range(len(pi)):
                s=0.
                for j in range(indptr[c],indptr[c+1]):
                    s+=weights[j]*factor[indices[j]]*pi[indices[j]]
                residual=max(residual,abs(pi[c]-s*(2-s)))
            return pi,residual,iteration+1
    raise RuntimeError('Survival Gauss-Seidel did not converge')


@njit(cache=True)
def site_factor(n,K,indptr,indices,weights):
    sf=np.ones(len(K))
    for d in range(len(K)):
        lam=0.
        for j in range(indptr[d],indptr[d+1]): lam+=2*weights[j]*n[indices[j]]/K[d]
        if lam>1e-12: sf[d]=-np.expm1(-lam)/lam
    return sf


class FastStepper(base.InheritedRegionalStepper):
    def establishment(self, phi, which, **kwargs):
        m=self.m
        n=m.K*phi[np.arange(m.M),m.genotypes]
        sf=site_factor(n,m.K,m.W.indptr,m.W.indices,m.W.data)
        if self._cached_factor is None or not np.array_equal(sf,self._cached_factor):
            viable_pi=np.zeros(m.M)
            for component,(ix,q) in enumerate(zip(m.components,m.component_matrices)):
                factor=sf[ix]; saved=self._component_cache[component]
                if saved is not None and np.array_equal(factor,saved[0]):
                    viable_pi[ix]=saved[1]; continue
                row=2*np.asarray(q@factor).ravel()
                if row.max()<=1+1e-11: pi=np.zeros(len(ix))
                else:
                    # A positive diagonal is itself a lower bound on rho(A).
                    certified=max(row.min(),float(np.max(2*q.diagonal()*factor)))>1+1e-11
                    if not certified:
                        if len(ix)==1: radius=float(row[0])
                        else:
                            d=sp.diags(np.sqrt(factor))
                            radius=float(eigsh(2*(d@q@d),k=1,which='LA',v0=np.ones(len(ix)),
                                tol=1e-11,maxiter=5000,return_eigenvectors=False)[0])
                        certified=radius>1+1e-11
                    if certified:
                        pi,error,_=survival_gs(q.indptr,q.indices,q.data,factor)
                        if error>1e-11: raise RuntimeError('Survival residual exceeds certification tolerance')
                        self.maximum_residual=max(self.maximum_residual,float(error))
                    else: pi=np.zeros(len(ix))
                viable_pi[ix]=pi; self._component_cache[component]=(factor.copy(),pi.copy())
            full=np.zeros((m.M,m.T));full[np.arange(m.M),m.genotypes]=viable_pi
            self._cached_factor=sf.copy(); self._cached_pi=full; self.solver_calls+=1
        return self._cached_pi[:,np.asarray(which)].copy()


class FastTree(base.MentorTree):
    def run_branch(self,n,events,seed,jump=None,max_updates=20000):
        original=base.InheritedRegionalStepper
        base.InheritedRegionalStepper=FastStepper
        try: return super().run_branch(n,events,seed,jump,max_updates)
        finally: base.InheritedRegionalStepper=original


