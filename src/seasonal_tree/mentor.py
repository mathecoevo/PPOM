"""Mentor regional-presence model on the existing five-tip effective-time tree.

The supplied Hybrid/RegionalHybrid classes are imported unchanged. Equation (1)
of ecotype_model2 supplies the missing seasonal landscape protocol. The tree
adapter preserves established populations at splits and stops at first-hit
colonisation-event counts. Fixed-n/lottery demography is retained, and differs
from the old capped-Poisson simulator. No ancestor truth enters this module.
"""
import os
for option in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[option] = '1'
from pathlib import Path
import importlib.util
import hashlib
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, spsolve
from numba import njit

from .vendor.ecotype_hybrid import Hybrid
from .vendor.ecotype_regional import RegionalHybrid
from . import tree_base as _old
load_public, build_W = _old.load_public, _old.build_W
EDGE_NAMES, TIP_EDGES = _old.EDGE_NAMES, _old.TIP_EDGES


@njit(cache=True)
def collapsed_map(phi, K, indptr, indices, gamma):
    out = np.zeros_like(phi)
    for d in range(len(phi)):
        log_empty = 0.
        for j in range(indptr[d], indptr[d+1]):
            log_empty += K[indices[j]] * np.log1p(-gamma[j] * phi[indices[j]])
        out[d] = -np.expm1(log_empty)
    return out


class HardSeasonalLandscape:
    """Equation (1), specialized to the tree's non-overlapping four ecotypes."""
    def __init__(self, data, n=2, mu=1e-7, dispersal=.02):
        self.K = data['K'].copy(); self.M = len(self.K); self.T = 4
        self.n = int(n); self.genotypes = data['state_genotypes'].copy()
        self.P = np.eye(self.T)[self.genotypes]
        self.Mu = np.zeros((self.T, self.T))
        for k in range(self.T):
            self.Mu[k,k] = 1-mu
            nearby = [l for l in (k-1,k+1) if 0 <= l < self.T]
            for l in nearby: self.Mu[k,l] = mu/len(nearby)
        cells = data['cells']; lookup = {tuple(c):i for i,c in enumerate(cells)}
        source, destination, probability = [], [], []
        for c,(r,col) in enumerate(cells):
            source.append(c); destination.append(c); probability.append(1-dispersal)
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                d = lookup.get((r+dr,col+dc))
                if d is not None:
                    source.append(c); destination.append(d); probability.append(dispersal/4)
        self.src = np.asarray(source); self.dst = np.asarray(destination)
        self.q = np.asarray(probability); self.diag = self.src == self.dst
        self.Q = sp.csr_matrix((self.q,(self.src,self.dst)),shape=(self.M,self.M))
        self.W = build_W(data, m=dispersal, mu=mu)
        d = np.repeat(np.arange(self.M), np.diff(self.W.indptr))
        self.gamma = -np.expm1(self.n*np.log1p(-self.W.data/self.K[d]))
        same = self.genotypes[self.src] == self.genotypes[self.dst]
        self.clonal = sp.csr_matrix((self.q[same]*(1-mu),
            (self.src[same],self.dst[same])),shape=(self.M,self.M))
        _, labels = connected_components(self.clonal, directed=False)
        self.components = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        self.component_matrices = [self.clonal[ix][:,ix].tocsr() for ix in self.components]
        self.map_calls = 0

    def phi(self, abundance):
        out = np.zeros((self.M,self.T))
        out[np.arange(self.M),self.genotypes] = np.asarray(abundance)/self.K
        return out

    def counts(self, phi):
        incompatible = phi.copy(); incompatible[np.arange(self.M),self.genotypes] = 0
        if np.max(np.abs(incompatible)) > 1e-12:
            raise ValueError('Incompatible ecotype in hard-selection state')
        return self.K*phi[np.arange(self.M),self.genotypes]

    def seasonal_map(self, phi):
        self.map_calls += 1
        viable = phi[np.arange(self.M),self.genotypes]
        out = collapsed_map(viable,self.K,self.W.indptr,self.W.indices,self.gamma)
        return self.phi(self.K*out)


class CertifiedRegionalHybrid(RegionalHybrid):
    """Same branching PGF; certify criticality and solve the positive fixed point.

Only viable source cells are needed under hard temperature selection. The
clonal dispersal graph is symmetric, allowing a symmetric similarity transform
for its reproduction spectral radius. Critical/subcritical components have
pi=0 exactly; a sparse Newton solve handles supercritical components.
"""
    def __init__(self, mod, kind, rng, region_of, **kwargs):
        if kind != 'seasonal': raise ValueError('This adapter supports seasonal branches only')
        super().__init__(mod,kind,rng,region_of,**kwargs)
        self._cached_factor = None; self._cached_pi = None
        self._component_cache = [None]*len(mod.components)
        self.solver_calls = 0; self.maximum_residual = 0.

    def establishment(self, phi, which, iters=500, tol=1e-7, x0=None):
        sf,_ = self._site_factor(phi)
        if self._cached_factor is None or not np.array_equal(sf,self._cached_factor):
            m = self.m; viable_pi = np.zeros(m.M)
            for component,(ix, q) in enumerate(zip(m.components,m.component_matrices)):
                factor = sf[ix]
                saved = self._component_cache[component]
                if saved is not None and np.array_equal(factor,saved[0]):
                    viable_pi[ix] = saved[1]
                    continue
                a = q.multiply(factor[None,:]).tocsr()
                upper = m.n*np.asarray(a.sum(1)).ravel().max()
                if upper <= 1+1e-11:
                    self._component_cache[component] = (factor.copy(),np.zeros(len(ix)))
                    continue
                lower = m.n*np.asarray(a.sum(1)).ravel().min()
                if lower > 1+1e-11:
                    radius = lower
                elif len(ix) == 1:
                    radius = m.n*a[0,0]
                else:
                    d = sp.diags(np.sqrt(factor))
                    symmetric = m.n*(d@q@d)
                    radius = float(eigsh(symmetric,k=1,which='LA',v0=np.ones(len(ix)),
                        tol=1e-11,maxiter=5000,return_eigenvectors=False)[0])
                if radius <= 1+1e-11:
                    self._component_cache[component] = (factor.copy(),np.zeros(len(ix)))
                    continue
                pi = np.ones(len(ix))
                for _ in range(6): pi = offspring_success(a@pi,m.n)
                identity = sp.eye(len(ix),format='csr')
                for iteration in range(100):
                    s = a@pi
                    value = offspring_success(s,m.n)
                    residual = pi-value; error = np.max(np.abs(residual))
                    if error < 1e-11: break
                    jac = identity-sp.diags(m.n*(1-s)**(m.n-1))@a
                    correction = spsolve(jac.tocsc(),residual)
                    scale = 1.
                    for _ in range(40):
                        trial = pi-scale*correction
                        if np.all((trial > 0)&(trial <= 1)):
                            tr = trial-offspring_success(a@trial,m.n)
                            if np.max(np.abs(tr)) < error: break
                        scale *= .5
                    else: raise RuntimeError('Establishment Newton line search failed')
                    pi = trial
                else: raise RuntimeError('Establishment fixed point did not converge')
                error = float(np.max(np.abs(pi-offspring_success(a@pi,m.n))))
                self.maximum_residual = max(self.maximum_residual,error)
                if error > 1e-10 or not np.any(pi > 0):
                    raise RuntimeError('Uncertified establishment probability')
                viable_pi[ix] = pi
                self._component_cache[component] = (factor.copy(),pi.copy())
            full = np.zeros((m.M,m.T)); full[np.arange(m.M),m.genotypes] = viable_pi
            self._cached_factor = sf.copy(); self._cached_pi = full
            self.solver_calls += 1
        return self._cached_pi[:,np.asarray(which)].copy()


def offspring_success(s,n):
    """1-(1-s)^n, including certain offspring survival without log warnings."""
    s = np.clip(s,0,1)
    if n == 2: return s*(2-s)
    with np.errstate(divide='ignore'):
        return -np.expm1(n*np.log1p(-s))


class InheritedRegionalStepper(CertifiedRegionalHybrid):
    """Persistent one-season form of the supplied RegionalHybrid.run loop.

Existing branch-start populations are already established; no new standing-
variation sample is taken. Births, arrival gating, seeding and regional loss
follow the supplied run loop. Region resolution is one cell on event-clock runs.
"""
    def start(self, abundance):
        self.phi_state = self.m.phi(abundance)
        self.present = (self.Rmat@(self.m.K[:,None]*self.phi_state)) > 0
        self.time = 0

    def prepare(self):
        phi = self.phi_state
        pi = self.establishment(phi,np.arange(self.m.T),tol=1e-4,iters=200)
        new = self.m.seasonal_map(phi)
        absent = ~self.present[self.reg]
        rate = np.where(absent,self.m.K[:,None]*new*pi,0.)
        lam = np.asarray(self.Rmat@rate)
        resident = np.where(absent,0.,new)
        return resident,rate,lam,pi

    def apply(self, prepared, hit=None):
        new,rate,lam,pi = prepared
        new = new.copy()
        if hit is None:
            hit = (lam > 0)&(self.rng.random(lam.shape) < -np.expm1(-lam))
        for r,l in zip(*np.nonzero(hit)):
            cells = self.cells_of[r]; w = rate[cells,l]
            cell = self.rng.choice(cells,p=w/w.sum())
            self._seed(new,cell,l,1/max(pi[cell,l],1e-3))
            self.present[r,l] = True
        self.phi_state = np.clip(new,0,1)
        gone = self.present&((self.Rmat@(self.m.K[:,None]*self.phi_state)) < self.absent_below)
        if gone.any():
            self.present &= ~gone
            self.phi_state *= self.present[self.reg]
        self.time += 1
        return self.m.counts(self.phi_state),hit


def conditional_hits(lam,rng):
    """Independent Poisson-hit indicators conditional on at least one hit."""
    total = float(lam.sum())
    if total <= 0: raise ValueError('No arrival hazard')
    if total > 1:
        count = 0
        while count == 0: count = rng.poisson(total)
    else:
        u = rng.random()*(-np.expm1(-total))
        count = 1; probability = np.exp(-total)*total; cumulative = probability
        while u > cumulative:
            count += 1; probability *= total/count; cumulative += probability
    selected = rng.choice(lam.size,size=count,p=lam.ravel()/total)
    hit = np.zeros(lam.size,dtype=bool); hit[selected] = True
    return hit.reshape(lam.shape)


class MentorTree(_old.RepairedTree):
    def __init__(self,history=0,mu=1e-7,jump=True,data=None):
        self.data = load_public(history=history,clock='effective') if data is None else data
        self.K = self.data['K']; self.mu = mu; self.jump = jump
        self.landscape = HardSeasonalLandscape(self.data,mu=mu)
        self.W = self.landscape.W
        cells = self.data['cells']; lookup = {tuple(c):i for i,c in enumerate(cells)}
        self.neighbours = np.full((len(cells),4),-1,dtype=np.int64)
        for j,(r,c) in enumerate(cells):
            for a,(dr,dc) in enumerate(((-1,0),(1,0),(0,-1),(0,1))):
                self.neighbours[j,a] = lookup.get((r+dr,c+dc),-1)
        self.order = np.lexsort((cells[:,1],cells[:,0])).astype(np.int64)
        self.branch_records = []

    def run_branch(self,n,events,seed,jump=None,max_updates=20000):
        n = np.asarray(n,dtype=float).copy(); ever = n > 0; ticks = 0
        if events == 0: return n,np.array([0,0,0,0,0])
        if events > len(n)-ever.sum(): return n,np.array([0,0,0,0,1])
        rng = np.random.default_rng(int(seed))
        hybrid = InheritedRegionalStepper(self.landscape,'seasonal',rng,np.arange(len(n)))
        hybrid.start(n)
        updates = skipped = jumps = stable = 0
        use_jump = self.jump if jump is None else jump
        while ticks < events:
            if n.sum() == 0 or ever.all(): return n,np.array([ticks,updates,skipped,jumps,1])
            if updates >= max_updates:
                raise RuntimeError('Mentor branch update limit; never converted to zero likelihood')
            prepared = hybrid.prepare(); resident,rate,lam,pi = prepared
            predicted = self.landscape.counts(resident)
            possible_loss = np.any((predicted>0)&(predicted<1))
            stationary = np.max(abs(predicted-n)) < 1e-9 and not possible_loss
            if use_jump and stable >= 10 and stationary:
                total = float(lam.sum())
                if total <= 0: return n,np.array([ticks,updates,skipped,jumps,1])
                waiting = int(rng.geometric(-np.expm1(-total)))
                if waiting < 1: raise RuntimeError('Invalid geometric waiting time')
                skipped += waiting; jumps += 1
                n,_ = hybrid.apply(prepared,conditional_hits(lam,rng))
            else:
                n,_ = hybrid.apply(prepared); updates += 1
            gained = (n>0)&~ever
            if gained.any(): ever |= gained; ticks += 1; stable = 0
            else: stable = stable+1 if stationary else 0
        self.branch_records.append(dict(events=int(events),ordinary_updates=updates,
            skipped_seasons=skipped,arrival_jumps=jumps,pi_solves=hybrid.solver_calls,
            maximum_pi_residual=hybrid.maximum_residual))
        return n,np.array([ticks,updates,skipped,jumps,0])

    def manifest(self):
        return dict(model='Mentor regional-presence seasonal hybrid; fixed-n/lottery',
            supplied_core_imported=['ecotype_hybrid.py','ecotype_regional.py'],
            supplied_sources_unmodified=False, supplied_change="relative imports only in vendor/ecotype_regional.py",missing_base_protocol='Equation (1) reconstructed under hard selection',
            offspring_per_breeder=2,offspring_distribution='fixed number, not old Poisson(2)',
            dispersal=.02,mutation=self.mu,environment_known=True,
            pi_solver='Same fixed-n branching PGF; spectral classification and sparse Newton',
            pi_floor_for_seeding=1e-3,extinction_threshold=1.,regional_resolution='one valid cell per region',
            root_prior='original uniform nonempty subsets of 27 public regions',
            initial_root_abundance='K in selected regions; post-burn-in state',
            inherited_absolute_abundance=True,inherited_established_state=True,
            standing_variation_resampled_at_each_branch=False,
            branch_clock='first-colonisation event-generations; first-hit endpoint',
            effective_branch_counts=dict(zip(EDGE_NAMES,map(int,self.data['times']))),
            stationary_geometric_waiting=self.jump,stationary_freezing_is_approximation=True,
            public_inputs_only=True,internal_pseudo_observations=False,
            sha256_adapter=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


# Compatibility with the already validated whole-history likelihood and sampler.
RepairedTree = MentorTree
