# src/qed.py
import numpy as np
from typing import Tuple, List
"""
Quantization / Edge Determination utilities (QED)
------------------------------------------------
The functions in this file are used to:
 - Build histograms for each dimension (used to determine bin edges for quantization).
 - Select bin windows that depend on the query (query-dependent bins).
 - Check whether a point lies within the selected bin window.
 - Calculate the preliminary quantization score between a point and the query.

Note:
 - Use numpy for performance; the functions assume that the data fits in memory.
 - build_histograms returns `edges` with shape (D, n_bins+1) similar to the output of np.histogram.
"""

def build_histograms(data: np.ndarray, n_bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-dimension histogram edges and counts.
    data: (N, D)
    returns:
        edges  -- ndarray of shape (D, n_bins + 1)
                  The bin boundaries (edges) for each dimension.
        counts -- ndarray of shape (D, n_bins)
                  The number of samples that fall into each bin per dimension.
    """
    N, D = data.shape
    # Initialize an empty array with D entry, each entry have n+1 edges
    edges = np.zeros((D, n_bins + 1), dtype=float)
    # Initialize an empty array with D entry, each entry have n count
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
    """
    Check if a given point falls inside the selected bin window across all dimensions.
    """
    for d in range(point.shape[0]):
        pbin = np.searchsorted(edges[d], point[d], side='right') - 1
        lo, hi = sel_bins[d]
        if not (lo <= pbin <= hi):
            return False
    return True

def quantify_score(point: np.ndarray, query: np.ndarray, edges: np.ndarray) -> float:
    """
    Compute a simple heuristic score between a point and the query.

    Current implementation:
        - Returns negative L2 distance: -||point - query||_2
          (so higher score = more similar).
    """
    return -np.linalg.norm(point - query)