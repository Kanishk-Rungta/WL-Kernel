"""Optimized graph-local maps, separate from the frozen Phase 1 implementation."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.sparse import csr_matrix, triu, save_npz, load_npz
from .feature_extraction import directions
from .swwl_kernel import squared_distances

VERSION = 'phase2-v1'
STYLES = {'formula': ('linear', 'squared_euclidean', 0.),
          'authors': ('lower', 'euclidean', 1e-7)}


@dataclass
class Structure:
    adjacency: csr_matrix
    degree: np.ndarray
    rows: np.ndarray
    gaps: np.ndarray
    edge_gaps: np.ndarray
    temperature: float
    operators: dict = field(default_factory=dict)


def structure(graph):
    a = graph.adjacency.tocsr()
    if (a != a.T).nnz or np.any(a.data != 1):
        raise ValueError('Phase 2 expects unweighted undirected graphs')
    degree = np.asarray(a.sum(axis=1)).ravel()
    rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
    gaps = np.abs(degree[rows] - degree[a.indices])
    # Include diagonal once if a caller supplies self-loops; ENZYMES has none.
    upper = triu(a, k=0).tocoo()
    edge_gaps = np.abs(degree[upper.row] - degree[upper.col])
    temperature = 1. + float(np.median(edge_gaps)) if len(edge_gaps) else 1.
    return Structure(a, degree, rows, gaps, edge_gaps, temperature)


def calibrated_temperature(graph):
    """No labels or dataset-level fitted statistics are accessed."""
    return structure(graph).temperature


def operator(s, aggregation='uniform', tau=None):
    if aggregation == 'self_calibrated':
        tau = s.temperature
    elif aggregation == 'degree_adaptive':
        if tau is None or np.isnan(tau) or tau <= 0:
            raise ValueError('tau must be positive')
    elif aggregation != 'uniform':
        raise ValueError(aggregation)
    key = (aggregation, tau)
    if key in s.operators:
        return s.operators[key]
    a = s.adjacency
    weights = np.empty(a.nnz)
    for v in range(a.shape[0]):
        start, end = a.indptr[v:v+2]
        if start == end:
            continue
        if aggregation == 'uniform' or np.isinf(tau):
            score = np.zeros(end-start)
        else:
            # Same stable softmax as Phase 1 for ordinary temperatures;
            # subtract before dividing only for extreme underflow-scale tau.
            gaps = s.gaps[start:end]
            with np.errstate(over='ignore', invalid='ignore'):
                score = -gaps / tau
                if not np.isfinite(score.max()):
                    score = -(gaps - gaps.min()) / tau
        w = np.exp(score-score.max())
        weights[start:end] = w/w.sum()
    result = csr_matrix((weights, a.indices.copy(), a.indptr.copy()), shape=a.shape)
    s.operators[key] = result
    return result


def history(graph, s, op, max_h, features='original'):
    if max_h < 0 or features not in ('original', 'degree_augmented'):
        raise ValueError('Invalid refinement configuration')
    x = graph.x.copy()
    if features == 'degree_augmented':
        x = np.column_stack([x, np.log1p(s.degree)])
    values = [x]
    isolates = s.degree == 0
    for _ in range(max_h):
        new = .5*(x + op@x)
        new[isolates] = x[isolates]
        values.append(new)
        x = new
    return values


def project(histories, H, P=20, Q=20, seed=42, kernel_style='formula'):
    if H < 0 or P < 1 or Q < 2:
        raise ValueError('H>=0, P>=1, Q>=2 required')
    interpolation = STYLES[kernel_style][0]
    d = histories[0][0].shape[1]*(H+1)
    projection = directions(d, P, seed)
    quantiles = np.linspace(0, 1, Q)
    return np.stack([
        np.quantile(np.concatenate(values[:H+1], axis=1)@projection,
                    quantiles, axis=0, method=interpolation).T.ravel()/np.sqrt(P*Q)
        for values in histories])


def distances(phi, kernel_style='formula'):
    d = squared_distances(phi)
    return d if STYLES[kernel_style][1] == 'squared_euclidean' else np.sqrt(d)


def kernel(d, gamma, kernel_style='formula'):
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    K = np.exp(-gamma*d)
    # Exact nugget convention found in official experiments/config.yml.
    nugget = STYLES[kernel_style][2]
    if nugget:
        K.flat[::len(K)+1] += nugget
    return K


def graph_fingerprint(graphs):
    digest = hashlib.sha256()
    for g in graphs:
        for a in [g.x, g.adjacency.indptr, g.adjacency.indices, g.adjacency.data]:
            a = np.ascontiguousarray(a)
            digest.update(str((a.shape, a.dtype.str)).encode())
            digest.update(a.tobytes())
    return digest.hexdigest()


def cache_spec(dataset, method, features, H, tau, seed, P, Q, style):
    interpolation, distance, nugget = STYLES[style]
    return dict(version=VERSION, dataset=dataset, method=method, features=features,
                H=H, temperature=tau, calibration='1+median_undirected_degree_gap',
                seed=seed, P=P, Q=Q, style=style, interpolation=interpolation,
                distance=distance, nugget=nugget, precision='float64',
                rng='numpy-PCG64-SeedSequence(seed,dimension,P)')


class Cache:
    """Content-addressed maps/distances/operators; callers choose cold or warm use."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, spec):
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def save(self, spec, **arrays):
        key = self.key(spec)
        np.savez(self.directory/f'{key}.npz', **arrays)
        (self.directory/f'{key}.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')

    def load(self, spec):
        path = self.directory/f'{self.key(spec)}.npz'
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def save_operator(self, spec, index, op):
        save_npz(self.directory/f'{self.key(spec)}_graph{index}.npz', op, compressed=False)

    def load_operator(self, spec, index):
        return load_npz(self.directory/f'{self.key(spec)}_graph{index}.npz')
