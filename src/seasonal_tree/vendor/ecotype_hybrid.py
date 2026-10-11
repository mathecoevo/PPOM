"""
Hybrid mean-field / branching-process model for rare ecotypes.

Ecotypes are either PRESENT (followed by the mean field) or ABSENT (held at zero).
  * At the start, an ecotype is absent if its expected number of individuals over the whole
    landscape is below `absent_below` (default 1). Standing variation in such ecotypes is sampled:
    the number of individuals is Poisson with the mean-field expectation, each placed in a cell in
    proportion to its expected numbers, and each founds a lineage that survives with probability pi
    (computed in the new environment). If any survives, the ecotype becomes present.
  * Each generation, the mean field says how many new breeders of each absent ecotype l would be
    produced by mutation in each cell d: m[d, l] (expected number per season, or rate per unit time).
  * Each such new breeder founds a lineage that survives with probability pi[d, l], from a
    multi-type branching process for a single ecotype-l breeder in the current resident background.
  * Established lineages arrive as a Poisson process with rate sum_d m[d, l] pi[d, l]. On arrival the
    ecotype becomes present, seeded in the chosen cell with 1/pi individuals (the expected size of a
    lineage conditioned on survival), and from then on follows the mean field.
  * A present ecotype whose landscape total falls below `absent_below` becomes absent again.
  * `start_below` (default = absent_below): at the start, ecotypes with an expected landscape total
    below this are sampled as stochastic founders rather than treated deterministically.

Branching process (one generation = one breeder's reproduction). A breeder of ecotype l in cell c
produces n offspring; each independently becomes a breeder in cell d with probability
    w[c, d] = Q_cd * Mu_ll * P_dl * s_d
where s_d is the chance that a surviving offspring obtains a site:
    seasonal:   s_d = (1 - exp(-Lam_d)) / Lam_d     (lottery against Lam_d resident survivors, Poisson)
    continuous: s_d = free fraction of sites in d  (own vacated site always free)
The extinction probabilities x_c solve x_c = (1 - sum_d w[c, d] (1 - x_d))^n; pi = 1 - x.
"""
import numpy as np


class Hybrid:
    def __init__(self, mod, kind, rng, absent_below=1.0, start_below=None, dt=0.02, pi_every=0.5):
        self.m, self.kind, self.rng = mod, kind, rng
        self.absent_below, self.dt, self.pi_every = absent_below, dt, pi_every
        self.start_below = absent_below if start_below is None else start_below
        s, d = mod.src, mod.dst
        self.Mu_ll = np.diag(mod.Mu)                                   # (T,)
        self.qP = mod.q[:, None] * mod.P[d] * self.Mu_ll[None, :]       # (E,T): Q_cd Mu_ll P_dl
        import scipy.sparse as sp
        self.Src = sp.csr_matrix((np.ones(len(s)), (s, np.arange(len(s)))), shape=(mod.M, len(s)))

    # ---------------- branching-process establishment probability ----------------
    def _site_factor(self, phi):
        m = self.m
        if self.kind == "seasonal":
            A = np.zeros((m.M, m.T))
            np.add.at(A, m.dst, (m.q * m.K[m.src])[:, None] * (phi[m.src] @ m.Mu))
            Lam = (m.n * m.P / m.K[:, None] * A).sum(1)
            return np.where(Lam > 1e-12, -np.expm1(-Lam) / np.maximum(Lam, 1e-12), 1.0), None
        free = np.clip(1 - phi.sum(1), 0, 1)
        own = 1 / m.K + (1 - 1 / m.K) * free                            # landing in own cell
        return free, own

    def establishment(self, phi, which, iters=500, tol=1e-7, x0=None):  # noqa
        """Survival probability of a lineage founded by one breeder, per cell. x0: previous extinction
        probabilities to start from (capped below 1 so that iteration still finds the minimal root)."""
        m = self.m; s, d = m.src, m.dst
        sf, own = self._site_factor(phi)
        fac = sf[d].copy()
        if own is not None:
            fac[m.diag] = own[d[m.diag]]
        w = self.qP[:, which] * fac[:, None]                           # (E, L)
        x = np.zeros((m.M, len(which))) if x0 is None else np.minimum(x0, 0.999)
        for _ in range(iters):
            acc = self.Src @ (w * (1 - x[d]))
            new = (1 - np.minimum(acc, 1)) ** m.n
            if np.abs(new - x).max() < tol:
                x = new; break
            x = new
        return 1 - x                                                   # pi[c, l]

    # ---------------- one run ----------------
    def run(self, phi0, times):
        m, K = self.m, self.m.K
        phi = phi0.copy()
        counts = K[:, None] * phi
        present = counts.sum(0) >= self.start_below
        rare = np.flatnonzero(~present)
        phi[:, rare] = 0
        out, arrivals = [], []
        if rare.size:                                               # stochastic standing variation
            pi0 = self.establishment(phi, rare)
            for j, l in enumerate(rare):
                tot = counts[:, l].sum()
                if tot <= 0:
                    continue
                N = self.rng.poisson(tot)
                if N == 0:
                    continue
                cells = self.rng.choice(m.M, size=N, p=counts[:, l] / tot)
                est = cells[self.rng.random(N) < pi0[cells, j]]
                for cell in np.unique(est):
                    k = np.sum(est == cell)
                    phi[cell, l] = min(k / max(pi0[cell, j], 1e-3), K[cell]) / K[cell]
                    others = phi[cell].sum() - phi[cell, l]
                    if others + phi[cell, l] > 1:
                        keep = np.arange(m.T) != l
                        phi[cell, keep] *= (1 - phi[cell, l]) / others
                if est.size:
                    present[l] = True; arrivals.append((0.0, l, int(est[0])))
        t, next_pi = 0.0, 0.0
        pi_full = np.zeros((m.M, m.T)); have = np.zeros(m.T, bool)
        step = 1.0 if self.kind == "seasonal" else self.dt
        for T in times:
            while t < T - 1e-9:
                absent = np.flatnonzero(~present)
                if absent.size and (t >= next_pi - 1e-9 or not have[absent].all()):
                    pi_full[:, absent] = self.establishment(phi, absent); have[:] = False; have[absent] = True
                    next_pi = t + (1.0 if self.kind == "seasonal" else self.pi_every)
                # mean-field step (absent ecotypes included so that mutant production is computed)
                if self.kind == "seasonal":
                    new = m.seasonal_map(phi)
                    prod = K[:, None] * new                          # expected new breeders this season
                else:
                    rhs = m.continuous_rhs(phi)
                    new = phi + self.dt * rhs
                    prod = K[:, None] * np.maximum(rhs, 0) * self.dt  # expected new breeders in dt
                if absent.size:
                    rate = prod[:, absent] * pi_full[:, absent]        # established lineages expected
                    for j, l in enumerate(absent):
                        lam = rate[:, j].sum()
                        if lam > 0 and self.rng.random() < -np.expm1(-lam):
                            cell = self.rng.choice(m.M, p=rate[:, j] / lam)
                            seed = min(1 / max(pi_full[cell, l], 1e-3), K[cell])
                            others = new[cell].sum() - new[cell, l]
                            new[cell, l] = seed / K[cell]
                            if others + new[cell, l] > 1:             # make room if the cell is full
                                scale = (1 - new[cell, l]) / others
                                keep = np.arange(m.T) != l
                                new[cell, keep] *= scale
                            present[l] = True
                            arrivals.append((t + step, l, cell))
                    still = ~present
                    new[:, still] = 0
                phi = np.clip(new, 0, 1)
                gone = present & ((K[:, None] * phi).sum(0) < self.absent_below)
                phi[:, gone] = 0; present[gone] = False
                t += step
            out.append(phi.copy())
        return out, arrivals
