# qed_dask_demo.py
# Proof-of-concept: Query-dependent Equi-Depth quantization + distributed indexing using Dask

import numpy as np
import dask.array as da
from dask.distributed import Client, LocalCluster
from typing import Tuple, List

# -----------------------
# Utility: build synthetic dataset (for demo)
# -----------------------
def make_synthetic_data(n_samples=100_000, n_dims=100, chunksize=5_000, seed=42):
    rng = np.random.RandomState(seed)
    # Example: mixture of Gaussians for some structure
    centers = rng.normal(scale=5.0, size=(10, n_dims))
    labels = rng.randint(0, 10, size=n_samples)
    data = centers[labels] + rng.normal(scale=1.0, size=(n_samples, n_dims))
    darr = da.from_array(data, chunks=(chunksize, n_dims))
    return darr

# -----------------------
# Build distributed histograms per-dimension (global aggregated)
# We'll use consistent bin edges per dimension.
# -----------------------
def build_histograms(darr: da.Array, n_bins:int=1024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      bin_edges: array shape (D, n_bins+1)
      counts: array shape (D, n_bins) -> global counts per bin
    Implementation:
      - For efficiency we use same edges across dims computed from global min/max per dim.
      - Compute per-chunk histograms and sum them.
    """
    N, D = darr.shape
    # compute min/max per dimension (distributed)
    mins = darr.min(axis=0).compute()   # shape (D,)
    maxs = darr.max(axis=0).compute()   # shape (D,)

    # avoid zero-width
    maxs = np.where(maxs <= mins, mins + 1e-6, maxs)

    # prepare edges array (D, n_bins+1)
    bin_edges = np.stack([np.linspace(mins[d], maxs[d], n_bins+1) for d in range(D)], axis=0)

    # function to compute histograms per chunk (NumPy arrays)
    def chunk_hist(chunk):
        # chunk shape (chunk_rows, D)
        # We'll compute hist for each dimension and return flattened
        ch_counts = np.empty((D, n_bins), dtype=np.int64)
        for d in range(D):
            edges = bin_edges[d]
            cnts, _ = np.histogram(chunk[:, d], bins=edges)
            ch_counts[d] = cnts
        # return flattened so Dask can sum
        return ch_counts

    # Map across blocks with map_blocks: returns array (nblocks, D, n_bins) conceptually, but easier to compute with map_overlap? Simpler approach:
    # We'll compute hist on each block using .map_blocks with dtype int64 and shape (D, n_bins)
    sample_block = darr.to_delayed().ravel()[0]
    # Build a list of delayed histograms for each block
    from dask import delayed
    delayed_blocks = darr.to_delayed().ravel()
    delayed_hists = [delayed(chunk_hist)(db.compute()) for db in delayed_blocks]
    # aggregate
    total = delayed_hists[0]
    for dh in delayed_hists[1:]:
        total = delayed(lambda a,b: a + b)(total, dh)
    total_counts = total.compute()  # (D, n_bins)
    return bin_edges, total_counts

# -----------------------
# Given histograms, for a query q and fraction p, estimate per-dim thresholds around q
# -----------------------
def estimate_thresholds_from_hist(bin_edges: np.ndarray, counts: np.ndarray, q: np.ndarray, p: float):
    """
    For each dimension:
      - find the bin index where q[d] lies
      - expand left/right until cumulative count >= p * N
      - return low, high values (float)
    """
    D, n_bins = counts.shape
    N = counts.sum(axis=1)[0] if False else counts.sum() // D  # this is hacky; instead compute total N from sum of counts across dims / ??? 
    # Actually each dimension sum should equal total points
    N = counts.sum(axis=1)[0] if counts.shape[0] > 0 else 0
    # safer: compute N from sum of any row
    N = counts[0].sum()
    # thresholds arrays
    lows = np.empty(D, dtype=float)
    highs = np.empty(D, dtype=float)
    target = max(1, int(np.ceil(p * N)))
    for d in range(D):
        edges = bin_edges[d]
        cnts = counts[d]
        # find bin containing q[d]
        # note: if q outside edges, clamp
        if q[d] <= edges[0]:
            bidx = 0
        elif q[d] >= edges[-1]:
            bidx = n_bins - 1
        else:
            bidx = np.searchsorted(edges, q[d], side='right') - 1
            bidx = np.clip(bidx, 0, n_bins-1)
        # expand
        left = bidx
        right = bidx
        cum = cnts[bidx]
        while cum < target and (left > 0 or right < n_bins - 1):
            # expand towards side with larger neighboring count (greedy) — simple heuristic
            left_cnt = cnts[left-1] if left > 0 else -1
            right_cnt = cnts[right+1] if right < n_bins - 1 else -1
            if left_cnt >= right_cnt:
                left -= 1
                cum += cnts[left]
            else:
                right += 1
                cum += cnts[right]
        lows[d] = edges[left]
        highs[d] = edges[right+1]  # edges are n_bins+1
    return lows, highs

# -----------------------
# Query processing: distributed filter per-partition, return final top-k
# -----------------------
def query_qed(darr: da.Array, bin_edges: np.ndarray, counts: np.ndarray,
              q: np.ndarray, p: float=0.05, penalty: float=None, local_top_m: int=50, top_k:int=10):
    """
    Args:
      darr: dask array (N, D)
      bin_edges, counts: from build_histograms
      q: numpy array shape (D,)
      p: fraction per-dimension to consider "near"
      penalty: if None, set to a big value like mean range per dimension
      local_top_m: how many candidates to take from each partition
      top_k: final k
    Returns:
      list of (index, distance) top-k (indices are global indices approximated)
    """
    D = q.shape[0]
    # estimate thresholds
    lows, highs = estimate_thresholds_from_hist(bin_edges, counts, q, p)

    if penalty is None:
        # approximate scale: average per-dim range
        avg_range = np.mean([edges[-1] - edges[0] for edges in bin_edges])
        penalty = avg_range * D * 2.0

    # for each block, compute local top-m (we use delayed to operate in chunk)
    from dask import delayed
    delayed_blocks = darr.to_delayed().ravel()
    block_starts = np.cumsum([0] + [b.shape[0] for b in darr.chunks[0]])[:-1]  # start indices per block

    def process_block(block_arr, block_start):
        # block_arr: numpy array (nrows, D)
        nrows = block_arr.shape[0]
        approx_scores = np.zeros(nrows, dtype=float)
        # compute per-dim contributions
        for d in range(D):
            col = block_arr[:, d]
            in_mask = (col >= lows[d]) & (col <= highs[d])
            # contribution: abs difference if in mask else penalty
            approx_scores += np.where(in_mask, np.abs(col - q[d]), penalty)
        # get local top-m (smallest scores)
        idx_local = np.argpartition(approx_scores, min(len(approx_scores)-1, local_top_m-1))[:local_top_m]
        # return global indices and approx_scores and vectors for refine
        global_idx = block_start + idx_local
        return list(zip(global_idx.tolist(), approx_scores[idx_local].tolist(), block_arr[idx_local].tolist()))

    # build delayed calls and compute in parallel
    delayed_results = []
    for i, db in enumerate(delayed_blocks):
        bs = block_starts[i]
        delayed_results.append(delayed(process_block)(db.compute(), bs))
    # gather all candidates
    from dask import compute
    all_candidates = compute(*delayed_results)
    # flatten
    flat = [item for block in all_candidates for item in block]
    # sort by approx score and pick top M_global
    flat_sorted = sorted(flat, key=lambda x: x[1])
    M_global = min(len(flat_sorted), max(top_k*5, 100))  # gather more for safety
    top_candidates = flat_sorted[:M_global]

    # refine: compute exact Euclidean distances on the candidate set
    cand_indices = [c[0] for c in top_candidates]
    cand_vectors = np.array([c[2] for c in top_candidates])
    diffs = cand_vectors - q[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    final = sorted(zip(cand_indices, dists), key=lambda x: x[1])[:top_k]
    return final

# -----------------------
# Demo main
# -----------------------
if __name__ == "__main__":
    # start local cluster
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit="2GB")
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    print("Build dataset...")
    darr = make_synthetic_data(n_samples=50_000, n_dims=50, chunksize=5000)

    print("Building histograms (this may take some time)...")
    bin_edges, counts = build_histograms(darr, n_bins=256)
    print("Done histograms.")

    # sample a random query (or take one vector)
    q = darr[123].compute()  # example query

    print("Querying with QED filter...")
    final = query_qed(darr, bin_edges, counts, q, p=0.05, local_top_m=50, top_k=10)
    print("Top-k results (index, euclidean_distance):")
    for idx, dist in final:
        print(idx, dist)

    client.close()
