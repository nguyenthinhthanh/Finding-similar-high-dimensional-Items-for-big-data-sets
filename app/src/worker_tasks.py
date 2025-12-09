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

# Module-level worker state (one worker process -> one module instance)
WORKER_LSH_INDEX = None        # MinHashLSHIndex built on concatenated local shards
WORKER_LOCAL_DATA = None       # np.ndarray local_data (stacked shard arrays)
WORKER_INDEX_MAP = None        # list of tuples (shard_idx, start_offset, length)
WORKER_ASSIGNED_SHARDS = None  # list of (shard_idx, path) assigned to this worker

# ---------------------------------------------------------------------
# Helper: determine which shards belong to this worker (round-robin)
# ---------------------------------------------------------------------
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
    for i, path in assigned:
        print(f"    -> Assigned shard index={i}, path={path}", flush=True)

    return assigned

# ---------------------------------------------------------------------
# Initialization: build local LSH index from assigned shard files
# ---------------------------------------------------------------------
def build_local_lsh_init(worker_rank: int, n_workers: int, bands: int = 32, max_bucket_size: int = 5000):
    """
    Build the local MinHash LSH index for this worker based on assigned shards.
    This function is intended to be executed on the worker process at startup via client.run(..., workers=[addr]).
    It:
      - lists assigned shards
      - loads them and stacks into local_data
      - builds MinHashLSHIndex on local_data
      - stores index and mapping in module-level variables for subsequent queries
    Returns True on success.
    """
    global WORKER_LSH_INDEX, WORKER_LOCAL_DATA, WORKER_INDEX_MAP, WORKER_ASSIGNED_SHARDS

    assigned = list_local_shards(worker_rank, n_workers)
    WORKER_ASSIGNED_SHARDS = assigned

    arrays = []
    # List of (shard_idx, start_offset, length)
    index_map = []
    offset = 0
    for shard_idx, path in assigned:
        try:
            arr = np.load(path)
        except Exception as e:
            print(f"[Error] Failed to load shard {path}: {e}", flush=True)
            continue
        arrays.append(arr)
        length = arr.shape[0]
        index_map.append((shard_idx, offset, length))
        offset += length

    if len(arrays) == 0:
        WORKER_LOCAL_DATA = np.empty((0, 0), dtype=np.uint64)
        WORKER_LSH_INDEX = None
        WORKER_INDEX_MAP = []
        print("[Error] No local shards loaded; LSH index not built.", flush=True)
        return False

    # Stack into a single local array
    local_data = np.vstack(arrays).astype(np.uint64)
    WORKER_LOCAL_DATA = local_data
    WORKER_INDEX_MAP = index_map

    # build LSH index on local_data
    print(f"[Worker] Building LSH on local_data shape={local_data.shape} (bands={bands}) ...", flush=True)
    lsh_index = build_minhash_lsh_index(local_data, bands=bands, max_bucket_size=max_bucket_size, verbose=False)
    WORKER_LSH_INDEX = lsh_index

    print(f"[Worker] Built LSH index successfully: local_rows={local_data.shape[0]}, shards={len(assigned)}", flush=True)
    return True

# ---------------------------------------------------------------------
# Utility: convert local concatenated index -> (global_shard_idx, row_idx)
# ---------------------------------------------------------------------
def _local_idx_to_shard_row(local_idx: int):
    """
    Map local concatenated index -> (global_shard_idx, row_idx).
    Uses WORKER_INDEX_MAP which contains tuples (shard_idx, start_offset, length).
    """
    if WORKER_INDEX_MAP is None or len(WORKER_INDEX_MAP) == 0:
        raise RuntimeError("Index map not initialized on worker")
    # linear scan is fine because number of assigned shards per worker is typically small
    for shard_idx, start, length in WORKER_INDEX_MAP:
        if start <= local_idx < start + length:
            return shard_idx, int(local_idx - start)
    # not found
    raise IndexError(f"Local index {local_idx} not mapped to any shard")

# ---------------------------------------------------------------------
# Main query-time worker function (invoked remotely by the driver)
# ---------------------------------------------------------------------
def shard_qed_filter_local(query: np.ndarray, edges: np.ndarray, worker_rank: int, n_workers: int, top_m: int = 100) -> List[Tuple[Tuple[int,int], float, list]]:
    """
    Run on a worker; return top_m candidate tuples:
      ((global_shard_idx, row_idx), score, preview_list)

    Uses prebuilt WORKER_LSH_INDEX if available; otherwise falls back to scanning shards.
    """
    # If local LSH index is built, use it
    if WORKER_LSH_INDEX is not None:
        # Query LSH index (it expects signature shape (num_perm,) uint-like)
        try:
            ids, sims = WORKER_LSH_INDEX.query(query, k=top_m)
            print(f"[Worker Debug] Query result -> ids: {ids[:10]}, sims: {sims[:10]}", flush=True)
        except Exception as e:
            print(f"[Error] LSH query failed: {e}. Falling back to scanning.", flush=True)
            ids, sims = np.array([], dtype=int), np.array([], dtype=float)

        candidates = []
        for local_idx, score in zip(ids.tolist(), sims.tolist()):
            try:
                shard_idx, row_idx = _local_idx_to_shard_row(int(local_idx))
            except Exception as e:
                # Skip bad mapping
                print(f"[Error] Mapping failed for local_idx={local_idx}: {e}", flush=True)
                continue
            preview = WORKER_LOCAL_DATA[local_idx][:10].tolist() if WORKER_LOCAL_DATA.size else []
            candidates.append(((shard_idx, row_idx), float(score), preview))
        # Ensure sorted by score desc; return top_m
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_m]

    # --- Fallback: scan assigned shards and score each point (slower) ---
    print("[Error] No LSH index present, performing full scan on assigned shards...", flush=True)
    candidates = []
    shards = list_local_shards(worker_rank, n_workers)
    for si, shard_path in shards:
        arr = np.load(shard_path)
        for i, pt in enumerate(arr):
            s = quantify_score(pt, query, edges)
            preview = pt[:10].tolist()
            candidates.append(((si, i), float(s), preview))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_m]