"""
Regional-presence hybrid: presence of each ecotype is tracked separately in each region.

Regions are blocks of bx x by cells. For each region r and ecotype l the ecotype is either PRESENT in r
(followed by the mean field in the cells of r) or ABSENT from r (held at zero there).

  * Start. In each region, an ecotype with fewer than `start_below` expected individuals is not treated
    deterministically: the number of individuals is Poisson with the mean-field expectation, each is placed
    in a cell of the region in proportion to expected numbers, and each founds a lineage that survives with
    probability pi. If any survives, the ecotype is present in r, seeded with (survivors)/pi individuals.
  * Arrivals. Each step, the mean field gives the expected number m[c, l] of new ecotype-l breeders that
    would appear in cell c, whether by mutation or by immigration from regions where l is present. In a
    region where l is absent these become established lineages at rate sum_{c in r} m[c, l] pi[c, l]
    (Poisson); on arrival the ecotype becomes present in r, seeded with 1/pi individuals in the chosen cell.
  * Extinction. A present ecotype whose total in r falls below `absent_below` individuals becomes absent
    from r.
With a single region covering the landscape this reduces to the landscape-level hybrid.
"""
import numpy as np
import scipy.sparse as sp
from .ecotype_hybrid import Hybrid


def block_regions(nx, ny, bx, by):
    col = np.tile(np.arange(nx), ny); row = np.repeat(np.arange(ny), nx)
    nbx = int(np.ceil(nx / bx))
    return (row // by) * nbx + (col // bx)


class RegionalHybrid(Hybrid):
    def __init__(self, mod, kind, rng, region_of, absent_below=1.0, start_below=10.0, dt=0.02, pi_every=0.5):
        super().__init__(mod, kind, rng, absent_below=absent_below, start_below=start_below,
                         dt=dt, pi_every=pi_every)
        self.reg = np.asarray(region_of); self.R = self.reg.max() + 1
        self.Rmat = sp.csr_matrix((np.ones(mod.M), (self.reg, np.arange(mod.M))), shape=(self.R, mod.M))
        self.cells_of = [np.flatnonzero(self.reg == r) for r in range(self.R)]

    def _seed(self, phi, cell, l, n_ind):
        K = self.m.K
        phi[cell, l] = min(n_ind, K[cell]) / K[cell]
        others = phi[cell].sum() - phi[cell, l]
        if others + phi[cell, l] > 1 and others > 0:
            keep = np.arange(self.m.T) != l
            phi[cell, keep] *= (1 - phi[cell, l]) / others

    def run(self, phi0, times):
        m, K, rng = self.m, self.m.K, self.rng
        phi = phi0.copy()
        counts = K[:, None] * phi
        regtot = self.Rmat @ counts                                          # (R, T)
        pres = regtot >= self.start_below
        mask = pres[self.reg]
        phi = phi * mask
        # --- stochastic founders at the start ---
        pi = self.establishment(phi, np.arange(m.T)); self._x = 1 - pi
        for r, l in zip(*np.nonzero(~pres & (regtot > 0))):
            cells = self.cells_of[r]; w = counts[cells, l]; tot = w.sum()
            N = rng.poisson(tot)
            if N == 0:
                continue
            where = rng.choice(cells, size=N, p=w / tot)
            est = where[rng.random(N) < pi[where, l]]
            for cell in np.unique(est):
                self._seed(phi, cell, l, np.sum(est == cell) / max(pi[cell, l], 1e-3))
            if est.size:
                pres[r, l] = True
        out, arrivals = [], []
        t, next_pi = 0.0, 0.0
        step = 1.0 if self.kind == "seasonal" else self.dt
        for T in times:
            while t < T - 1e-9:
                if t >= next_pi - 1e-9:
                    pi = self.establishment(phi, np.arange(m.T), x0=self._x, tol=1e-4, iters=200); self._x = 1 - pi
                    next_pi = t + (1.0 if self.kind == "seasonal" else self.pi_every)
                if self.kind == "seasonal":
                    new = m.seasonal_map(phi); prod = K[:, None] * new
                else:
                    rhs = m.continuous_rhs(phi); new = phi + self.dt * rhs
                    prod = K[:, None] * np.maximum(rhs, 0) * self.dt
                absent = ~pres[self.reg]                                     # (M, T)
                rate = np.where(absent, prod * pi, 0.0)
                lamR = np.asarray(self.Rmat @ rate)                          # (R, T)
                hit = (lamR > 0) & (rng.random(lamR.shape) < -np.expm1(-lamR))
                new = np.where(absent, 0.0, new)
                for r, l in zip(*np.nonzero(hit)):
                    cells = self.cells_of[r]; w = rate[cells, l]
                    cell = rng.choice(cells, p=w / w.sum())
                    self._seed(new, cell, l, 1 / max(pi[cell, l], 1e-3))
                    pres[r, l] = True; arrivals.append((t + step, r, l, cell))
                phi = np.clip(new, 0, 1)
                regtot = self.Rmat @ (K[:, None] * phi)
                gone = pres & (regtot < self.absent_below)
                if gone.any():
                    pres &= ~gone
                    phi = phi * pres[self.reg]
                t += step
            out.append(phi.copy())
        return out, arrivals
