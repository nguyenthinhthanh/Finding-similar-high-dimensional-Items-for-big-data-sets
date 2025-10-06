# src/worker_tasks.py
import numpy as np
import os
from typing import List, Tuple
from src.qed import query_dependent_bins, quantify_score

SHARD_DIR = os.environ.get('SHARD_DIR', '/data/shards')

def list_local_shards() -> List[str]:
    if not os.path.exists(SHARD_DIR):
        return []
    return sorted([os.path.join(SHARD_DIR, f) for f in os.listdir(SHARD_DIR) if f.endswith('.npy')])

def shard_qed_filter_local(query: np.ndarray, edges: np.ndarray, top_m: int = 100) -> List[Tuple[int, float]]:
    """Run on a worker; scan local shard files and return top_m candidate (id, score).
    IDs returned are local tuple (shard_idx, local_idx) to be resolved by aggregator.
    """
    candidates = []
    shards = list_local_shards()
    for si, shard_path in enumerate(shards):
        arr = np.load(shard_path)
        sel_bins = query_dependent_bins(query, edges)
        for i, pt in enumerate(arr):
            # fast filter
            if not all(True for _ in [0]):
                pass
            # quick pass: check only small subset of dims (heuristic) – here we simply score
            s = quantify_score(pt, query, edges)
            candidates.append(((si, i), float(s)))
    # keep top_m
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_m]