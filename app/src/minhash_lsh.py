# app/src/minhash_lsh.py
import numpy as np
import time
import sys, os
import pandas as pd
from collections import defaultdict
"""
MinHash + LSH (banding) implementation for Jaccard similarity on documents.

Usage summary:
- Build MinHash signatures (num_perm)
- Build LSH index with banding (bands x rows)
- Query: returns candidates + estimated Jaccard similarity

Notes:
- Minimal external dependencies (only standard library + numpy)
- For production/speed, consider using `datasketch` library (optimized C code)
"""

# Build MinHash LSH index once (tune bands)
# Try 32, 16, 8 depending on recall/precision tradeoff
BANDS = 32

# ========== MinHash-aware search helpers ==========
class MinHashLSHIndex:
    """
    Build-once MinHash LSH index using banding.
    - data: np.ndarray shape (N, num_perm), dtype integer-like (uint64 or int)
    - bands: number of bands (bands * rows == num_perm)
    - max_bucket_size: cap per bucket to avoid pathological buckets
    """
    def __init__(self, data: np.ndarray, bands: int = 32, max_bucket_size: int = 5000):
        self.data = data
        self.N, self.num_perm = data.shape
        assert self.num_perm % bands == 0, "num_perm must be divisible by bands"
        self.bands = bands
        self.rows = self.num_perm // bands
        self.max_bucket_size = max_bucket_size
        self.tables = [defaultdict(list) for _ in range(self.bands)]
        self._build_tables()

    def _build_tables(self):
        """
        Build the LSH tables.
        For each data signature:
          - Split into bands
          - Convert each band to bytes (fast hashing)
          - Insert index into corresponding bucket (capped by max_bucket_size)
        """
        for idx in range(self.N):
            sig = self.data[idx]
            for b in range(self.bands):
                start = b * self.rows
                key = sig[start:start+self.rows].tobytes()
                tbl = self.tables[b]
                if len(tbl[key]) < self.max_bucket_size:
                    tbl[key].append(idx)

                # --- DEBUG: Check vector 1025 ---
                if idx == 1025:
                    print(f"[LSH DEBUG] Vector index={idx}, Band={b}", flush=True)
                    print(f"[LSH DEBUG] Signature preview (first 10 vals): {sig[:10]}", flush=True)
                    print(f"[LSH DEBUG] Type: {type(sig), sig.dtype}", flush=True)
                    print(f"[LSH DEBUG] Sub-signature for this band (sig[{start}:{start+self.rows}]): {sig[start:start+self.rows]}", flush=True)
                    print(f"[LSH DEBUG] Key (hex): {key.hex()[:40]}...", flush=True)
                    print(f"[LSH DEBUG] Position (start,end): {start, start+self.rows}", flush=True)
                    print(f"[LSH DEBUG] Current bucket size for this key: {len(tbl[key])}", flush=True)
                    print("\n", flush=True)

    def query(self, q: np.ndarray, k: int = 10, max_candidates: int = 2000, fallback_sample: int = 200):
        """
        Query a single signature q (1D array): returns (ids_array, sims_array)
        sims are estimated Jaccard = fraction of equal positions between q and candidate signature.
        """
        cand_set = set()
        for b in range(self.bands):
            start = b * self.rows
            key = q[start:start+self.rows].tobytes()
            bucket = self.tables[b].get(key)
            # --- DEBUG: print query band info ---
            print(f"[LSH DEBUG] Query Band={b}", flush=True)
            print(f"[LSH DEBUG] Query Signature preview (first 10 vals): {q[:10]}", flush=True)
            print(f"[LSH DEBUG] Type: {type(q), q.dtype}", flush=True)
            print(f"[LSH DEBUG] Sub-signature for this band (q[{start}:{start+self.rows}]): {q[start:start+self.rows]}", flush=True)
            print(f"[LSH DEBUG] Key (hex): {key.hex()[:40]}...", flush=True)
            print(f"[LSH DEBUG] Position (start,end): ({start}, {start+self.rows})", flush=True)
            print("\n", flush=True)
            if bucket:
                # print(f"Band {b}, key={key.hex()[:8]}..., bucket size={len(bucket)}")
                # for global_idx in bucket:
                #     shard_idx = global_idx // 5000
                #     row_idx = global_idx % 5000
                #     print(f"  global_idx={global_idx}, (shard={shard_idx}, row={row_idx})")
                cand_set.update(bucket)
            if len(cand_set) >= max_candidates:
                break

        if not cand_set:
            print("[LSH Debug] no candidates from buckets -> returning dummy (-1)", flush=True)
            cand_list = np.array([-1], dtype=int)
            sims = np.array([0.0], dtype=float)
            return cand_list, sims
        else:
            cand_list = np.fromiter(cand_set, dtype=int)
            print(f"[LSH Debug] cand_list size={cand_list.size}", flush=True)

        if cand_list.size == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # Shape (n_cand, num_perm)
        cand_sigs = self.data[cand_list]
        # Vectorized equality and mean -> Estimated Jaccard
        sims = (cand_sigs == q).mean(axis=1)
        # Indices into cand_list (descending)
        top_idxs = np.argsort(sims)[-k:][::-1]
        return cand_list[top_idxs], sims[top_idxs]
    
def minhash_lsh_search(queries, data, k=10, lsh_index: MinHashLSHIndex = None):
    """
    Prebuilt MinHashLSHIndex for each query.
    lsh_index must be built once and passed in (not None).
    """
    if lsh_index is None:
        raise ValueError("lsh_index must be provided to minhash_lsh_search_wrapper")
    all_results = []
    for q in queries:
        ids, sims = lsh_index.query(q, k=k)
        # if fewer than k, pad with random indices or leave as is (we'll return array rows of length k)
        if len(ids) < k:
            # fallback: pad with -1 to maintain shape, caller can handle if needed
            pad = np.full(k - len(ids), -1, dtype=int)
            ids = np.concatenate([ids, pad])
        all_results.append(ids[:k])
    return np.vstack(all_results)

def build_minhash_lsh_index(data, bands=BANDS, max_bucket_size=5000, verbose=True):
    """
    Build a MinHash LSH index from given data signatures.
    Returns
        lsh_index : MinHashLSHIndex
        A built LSH index ready for querying.
    """
    if verbose:
        print(f"Building MinHash-LSH index with bands={bands}, max_bucket_size={max_bucket_size}...", flush=True)
    
    # Create and build the index
    lsh_index = MinHashLSHIndex(data, bands=bands, max_bucket_size=max_bucket_size)
    
    if verbose:
        print(f"Built MinHashLSHIndex successfully: bands={bands}, rows={lsh_index.rows} \n", flush=True)
    return lsh_index