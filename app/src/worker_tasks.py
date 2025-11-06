# src/worker_tasks.py
import numpy as np
import sys, os
from typing import List, Tuple
from distributed import get_worker
"""
Worker-side Tasks for Dask-based Query Service
----------------------------------------------
This module contains functions executed on individual Dask workers.
It handles:
 - Listing local shard files for processing.
 - Scanning shard data and applying query-dependent filtering.
 - Scoring candidate points and returning top-k candidates per worker.

Notes:
 - Worker tasks return local IDs (shard index, row index) instead of global IDs.
 - Aggregation and top-k merging is handled by the query service.
"""

# Add current directory to Python import path (so local imports work)
sys.path.append(os.path.dirname(__file__))
from qed import query_dependent_bins, quantify_score
from minhash_lsh import build_minhash_lsh_index, minhash_lsh_search

SHARD_DIR = os.environ.get('SHARD_DIR', '/data/shards')

def list_local_shards(worker_rank: int, n_workers: int) -> List[Tuple[int, str]]:
    """
    Return a list of (global_shard_index, shard_path) assigned to THIS worker.
    Returns:
      list of (global_index, path) for shards assigned to this worker.
    """
    if not os.path.exists(SHARD_DIR):
        print(f"[ERROR] Shard directory not found: {SHARD_DIR}", flush=True)
        raise FileNotFoundError(f"Shard directory not found: {SHARD_DIR}")
    
    # Stable global list of shards
    all_files = sorted([f for f in os.listdir(SHARD_DIR) if f.endswith('.npy')])
    all_shards = [os.path.join(SHARD_DIR, f) for f in all_files]
    S = len(all_shards)

    # Build assigned list as (global_index, path)
    assigned = [
        (i, path)
        for i, path in enumerate(all_shards)
        if (i % n_workers) == worker_rank
    ]

    # Debug print so you can see partitioning on worker logs
    print(f"[Worker] Worker index={worker_rank}/{n_workers}, Total shards={S}, Assigned={len(assigned)}", flush=True)
    return assigned

def shard_qed_filter_local(query: np.ndarray, edges: np.ndarray, worker_rank: int, n_workers: int, top_m: int = 100) -> List[Tuple[int, float]]:
    """Run on a worker; scan local shard files and return top_m candidate (id, score).
    IDs returned are local tuple (shard_idx, local_idx) to be resolved by aggregator.
    """
    candidates = []
    # List of (global_index, path)
    shards = list_local_shards(worker_rank, n_workers)
    for si, shard_path in shards:
        arr = np.load(shard_path)
        # Bin index choosen (lo, hi) 
        sel_bins = query_dependent_bins(query, edges)
        for i, pt in enumerate(arr):
            # Fast filter
            if not all(True for _ in [0]):
                # Do nothing, need to code fast filter
                pass
            # Quick pass: check only small subset of dims (heuristic) – here we simply score
            s = quantify_score(pt, query, edges)
            # Create preview (first preview_len elements)
            preview = pt[:10].tolist()
            # Each candidate: ((shard_idx, row_idx), score, preview)
            candidates.append(((si, i), float(s), preview))
    # Keep top_m
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_m]