"""Public geometry and mass-preserving split helpers; no old Poisson propagation."""
import numpy as np
from numba import njit
from .data import load_public, build_W, EDGE_NAMES, TIP_EDGES

@njit(cache=True)
def positive_poisson(lam):
    if lam > 1:
        while True:
            n = np.random.poisson(lam)
            if n: return n
    # Avoid rejection and cancellation at very small mutation intensities.
    u = np.random.random()*(-np.expm1(-lam))
    p = np.exp(-lam)*lam; cumulative = p; n = 1
    while u > cumulative:
        n += 1; p *= lam/n; cumulative += p
    return n

@njit(cache=True)
def geometric_split(n, cells, neighbours, order):
    """Frozen geographic rule, adapted only from integer to expected abundance."""
    J = len(n); labels = np.full(J, -1, dtype=np.int64)
    component_mass = np.zeros(J); queue = np.empty(J, dtype=np.int64); number = 0
    for k in order:
        if n[k] <= 0 or labels[k] >= 0: continue
        labels[k] = number; queue[0] = k; read = 0; size = 1
        while read < size:
            i = queue[read]; read += 1; component_mass[number] += n[i]
            for d in neighbours[i]:
                if d >= 0 and n[d] > 0 and labels[d] < 0:
                    labels[d] = number; queue[size] = d; size += 1
        number += 1
    left = np.zeros(J); right = np.zeros(J)
    if number == 0: return left, right
    if number >= 2:
        # Stable sorting keeps the original row/column scan order for tied mass.
        ranked = np.argsort(-component_mass[:number], kind='mergesort')
        assigned = np.empty(number, dtype=np.int64); totals = np.zeros(2)
        for component in ranked:
            side = 0 if totals[0] <= totals[1] else 1
            assigned[component] = side; totals[side] += component_mass[component]
        for j in range(J):
            if labels[j] >= 0:
                if assigned[labels[j]] == 0: left[j] = n[j]
                else: right[j] = n[j]
    else:
        occupied = np.empty(J, dtype=np.int64); size = 0
        r = 0.; c = 0.
        for j in order:
            if n[j] > 0:
                occupied[size] = j; size += 1; r += cells[j, 0]; c += cells[j, 1]
        if size == 1:
            j = occupied[0]
            # Rare counts are integers; common expected counts may be fractional.
            integer = int(n[j]); a = np.random.binomial(integer, .5)
            left[j] = a+.5*(n[j]-integer); right[j] = n[j]-left[j]
        else:
            r /= size; c /= size; best = -1.; seed1 = occupied[0]
            for a in range(size):
                j = occupied[a]; distance = (cells[j,0]-r)**2+(cells[j,1]-c)**2
                if distance > best: best = distance; seed1 = j
            best = -1.; seed2 = occupied[0]
            for a in range(size):
                j = occupied[a]
                distance = np.sum((cells[j]-cells[seed1])**2)
                if distance > best: best = distance; seed2 = j
            for a in range(size):
                j = occupied[a]
                d1 = np.sum((cells[j]-cells[seed1])**2)
                d2 = np.sum((cells[j]-cells[seed2])**2)
                side = d2 < d1
                if d1 == d2: side = np.random.random() < .5
                if side: right[j] = n[j]
                else: left[j] = n[j]
    if left.sum() == 0 or right.sum() == 0:
        donor = right if left.sum() == 0 else left
        receiver = left if left.sum() == 0 else right
        # Frozen code scans genotype, then coordinates in this special fallback.
        for j in range(J):
            if donor[j] > 0:
                if donor[j] >= 2: donor[j] -= 1; receiver[j] += 1
                break
    return left, right

@njit(cache=True)
def seeded_split(n, seed, cells, neighbours, order):
    np.random.seed(seed)
    return geometric_split(n, cells, neighbours, order)

class RepairedTree:
    def root(self,mask,fraction=1.):
        return self.K*fraction*((int(mask)>>self.data['state_blocks'])&1)
