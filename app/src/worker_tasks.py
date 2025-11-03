# src/worker_tasks.py
import numpy as np
import sys, os
from typing import List, Tuple
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

SHARD_DIR = os.environ.get('SHARD_DIR', '/data/shards')

def list_local_shards() -> List[str]:
    """
    Return a sorted list of local shard file paths in SHARD_DIR.
    """
    if not os.path.exists(SHARD_DIR):
        raise FileNotFoundError(f"Shard directory not found: {SHARD_DIR}")
    return sorted([os.path.join(SHARD_DIR, f) for f in os.listdir(SHARD_DIR) if f.endswith('.npy')])

def shard_qed_filter_local(query: np.ndarray, edges: np.ndarray, top_m: int = 100) -> List[Tuple[int, float]]:
    """Run on a worker; scan local shard files and return top_m candidate (id, score).
    IDs returned are local tuple (shard_idx, local_idx) to be resolved by aggregator.
    """
    candidates = []
    shards = list_local_shards()
    for si, shard_path in enumerate(shards):
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
            # Each candidate: ((shard_idx, row_idx), score)
            candidates.append(((si, i), float(s)))
    # Keep top_m
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_m]