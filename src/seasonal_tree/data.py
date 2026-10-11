"""Portable public inputs; inference never loads evaluation-only material."""
from pathlib import Path
import hashlib
import numpy as np
import scipy.sparse as sp

EDGE_NAMES = ['A', 'B', 'node_AB', 'C', 'D', 'E', 'node_ABC', 'node_DE']
TIP_EDGES = np.array([0, 1, 3, 4, 5])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_public(path=None, history=0, clock='effective'):
    if path is None:
        raise ValueError('Supply a public.npz path; no machine-specific default is used')
    if history != 0 or clock != 'effective':
        raise ValueError('This release supports a supplied effective-clock five-tip dataset')
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in (
            'cells', 'state_blocks', 'state_genotypes', 'K', 'shape', 'times')}
        for k in ('global_indices', 'areas', 'tie_order', 'calendar_times'):
            if k in z.files:
                data[k] = z[k].copy()
    validate_public(data)
    data['shape'] = tuple(map(int, data['shape']))
    data.update(clock=clock, history=history,
                metadata_keys_read=tuple(data), public_sha256=sha256(path))
    return data


def validate_public(data):
    cells = np.asarray(data['cells'])
    k = np.asarray(data['K'])
    if cells.shape != (len(k), 2) or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError('cells must be an integer array with shape (M, 2)')
    if len(np.unique(cells, axis=0)) != len(k):
        raise ValueError('Duplicate valid cells')
    if not np.all(np.isfinite(k)) or not np.all(k >= 1):
        raise ValueError('Finite carrying capacities >= 1 are required')
    for key, limit in (('state_blocks', 27), ('state_genotypes', 4)):
        a = np.asarray(data[key])
        if a.shape != k.shape or not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f'{key} must be an integer vector of length M')
        if np.any((a < 0) | (a >= limit)):
            raise ValueError(f'{key} values outside 0..{limit - 1}')
    if set(np.unique(data['state_blocks'])) != set(range(27)):
        raise ValueError('This release requires all 27 root catalogue regions')
    shape = np.asarray(data['shape'])
    if shape.shape != (2,) or np.any(shape <= 0) or np.any(cells < 0) or np.any(cells >= shape):
        raise ValueError('Invalid map shape or cells outside map bounds')
    times = np.asarray(data['times'])
    if times.shape != (8,) or not np.issubdtype(times.dtype, np.integer) or np.any(times <= 0):
        raise ValueError('Eight strictly positive effective branch counts are required')


def mutation_probability(source, destination, mu):
    if source == destination:
        return 1 - mu
    if abs(source - destination) != 1:
        return 0.
    return mu if source in (0, 3) else mu / 2


def build_W(data, m=.02, mu=1e-7):
    cells, genotypes = data['cells'], data['state_genotypes']
    index = {tuple(cell): i for i, cell in enumerate(cells)}
    rows, cols, values = [], [], []
    for c, (r, col) in enumerate(cells):
        rows.append(c); cols.append(c); values.append((1 - m) * (1 - mu))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d = index.get((r + dr, col + dc))
            if d is not None:
                value = m / 4 * mutation_probability(genotypes[c], genotypes[d], mu)
                if value:
                    rows.append(d); cols.append(c); values.append(value)
    # Destinations in rows, sources in columns. Losses are not renormalized.
    return sp.csr_matrix((values, (rows, cols)), shape=(len(cells), len(cells)))


def load_observations(path, case, cells):
    with np.load(path, allow_pickle=False) as z:
        labels = list(map(str, z['case']))
        if case not in labels:
            raise ValueError(f'Unknown observation case: {case}')
        counts = z['counts'][labels.index(case)].copy()
    if counts.shape != (5, cells) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError('Counts must have shape (5, M) and integer dtype')
    if np.any(counts < 0) or counts.sum() == 0:
        raise ValueError('Nonnegative counts and at least one observation are required')
    return counts

