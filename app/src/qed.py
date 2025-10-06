# src/qed.py
import numpy as np
from typing import Tuple, List


def build_histograms(data: np.ndarray, n_bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-dimension histogram edges and counts.
    data: (N, D)
    returns: edges (D, n_bins+1), counts (D, n_bins)
    """
    N, D = data.shape
    edges = np.zeros((D, n_bins + 1), dtype=float)
    counts = np.zeros((D, n_bins), dtype=int)
    for d in range(D):
        hist, bin_edges = np.histogram(data[:, d], bins=n_bins)
        edges[d, :] = bin_edges
        counts[d, :] = hist
    return edges, counts

def query_dependent_bins(query: np.ndarray, edges: np.ndarray, p_fraction: float = 0.02) -> List[tuple]:
    """Return selected bin windows per-dimension for a query.
    Simple heuristic: pick +/- r bins around query's bin where r = max(1, int(p_fraction * n_bins)).
    """
    D = query.shape[0]
    n_bins = edges.shape[1] - 1
    sel = []
    for d in range(D):
        qv = query[d]
        bin_idx = np.searchsorted(edges[d], qv, side='right') - 1
        r = max(1, int(p_fraction * n_bins))
        lo = max(0, bin_idx - r)
        hi = min(n_bins - 1, bin_idx + r)
        sel.append((lo, hi))
    return sel

def point_passes_bins(point: np.ndarray, edges: np.ndarray, sel_bins: List[tuple]) -> bool:
    for d in range(point.shape[0]):
        pbin = np.searchsorted(edges[d], point[d], side='right') - 1
        lo, hi = sel_bins[d]
        if not (lo <= pbin <= hi):
            return False
    return True

def quantify_score(point: np.ndarray, query: np.ndarray, edges: np.ndarray) -> float:
    # simple score: negative L2 but penalize dimensions far from query bin
    return -np.linalg.norm(point - query)