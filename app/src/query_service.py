# src/query_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from distributed import Client
import numpy as np
import sys, os
import time
import logging
from typing import List, Tuple

# Add current directory to Python import path (so local imports work)
sys.path.append(os.path.dirname(__file__))

# -----------------------------------------------------------
# Dask-based Query Service
# -----------------------------------------------------------
# This service exposes a REST API (via FastAPI) that accepts
# vector search queries, distributes the computation to Dask
# workers, collects partial results, merges them, and returns
# the top-k most similar candidates.
# -----------------------------------------------------------

# Initialize FastAPI app
app = FastAPI()

# Get the Dask scheduler address from environment variable.
# This allows the service to connect to the distributed cluster.
DASK_ADDR = os.environ.get('DASK_SCHEDULER_ADDRESS', 'tcp://scheduler:8786')
client = Client(DASK_ADDR)

# -----------------------------------------------------------
# Request schema definition for /query endpoint
# -----------------------------------------------------------
class QueryRequest(BaseModel):
    """
    Represents the JSON request body schema for a query request.
    Used by FastAPI (via Pydantic) to automatically parse and validate input.

    Attributes:
        vector (List[float]): The query feature vector (e.g., embedding)
            that will be compared against database vectors for similarity.
        k (int): The number of top results (nearest neighbors) to return.
            Defaults to 10 if not provided.
    """
    vector: List[int]
    k: int = 10

# -----------------------------------------------------------
# Load precomputed histogram edges (for quantization bins)
# -----------------------------------------------------------
# The edges array is shared across all queries.
# It’s precomputed offline and stored at /data/hist_edges.npy.
# -----------------------------------------------------------
EDGES_PATH = os.environ.get('EDGES_PATH', '/data/hist_edges.npy')
if os.path.exists(EDGES_PATH):
    GLOBAL_EDGES = np.load(EDGES_PATH)
else:
    GLOBAL_EDGES = None
    raise FileNotFoundError(f"Precomputed edges file not found at {EDGES_PATH}")

# ------------------------------------------------------------------
# Helper: wait for expected Dask workers to appear
# ------------------------------------------------------------------
def wait_for_workers(client, timeout=30, poll_interval=0.5, expected_count=3):
    """
    Wait until all expected Dask workers are connected or timeout.
    """
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            sinfo = client.scheduler_info()
            workers = sinfo.get("workers", {})
            n = len(workers)
            if n >= expected_count:
                return workers
        except Exception as e:
            print(f"[Error] {e}", flush=True)
        time.sleep(poll_interval)
    print(f"[Error] Timeout reached: only {len(workers) if 'workers' in locals() else 0}/{expected_count} workers.", flush=True)
    return workers if 'workers' in locals() else {}

# ------------------------------------------------------------------
# FastAPI startup event: ensure workers present and request LSH build
# ------------------------------------------------------------------
@app.on_event("startup")    
def startup_event():
    """
    FastAPI lifecycle hook: called when the service starts.
    Ensures all Dask workers are reachable and have required modules loaded.
    """
    print("[Startup] Waiting for Dask workers (timeout=30s)...", flush=True)
    workers = wait_for_workers(client, timeout=30, poll_interval=0.5, expected_count=3)

    if not workers:
        print("[Startup] WARNING: No workers detected; continuing but queries may fail.", flush=True)
    else:
        print("[Startup] Workers detected:", list(workers.keys()), flush=True)

    if workers:
        try:
            responses = client.run(lambda: "worker ready")
            print("[Startup] Connected workers:", responses, flush=True)
            for addr, msg in responses.items():
                print(f"  - {addr}: {msg}", flush=True)
        except Exception as e:
            print(f"[Startup] client.run() failed during startup: {e}", flush=True)

    print("[Startup] Building local LSH indices on workers...", flush=True)
    # Choose bands and max_bucket_size consistent with index build settings
    BANDS = 32
    MAX_BUCKET = 5000
    import worker_tasks
    for wi, addr in enumerate(list(workers.keys())):
        try:
            # Eun build_local_lsh_init on specific worker addr with its rank
            client.run(worker_tasks.build_local_lsh_init,
                        wi, len(workers), BANDS, MAX_BUCKET,
                        workers=[addr])
            print(f"[Startup] Requested LSH build on worker {addr} (rank={wi})", flush=True)
        except Exception as e:
            print(f"[Error] Failed LSH build init on worker {addr}: {e}", flush=True)
    
    q = np.load("/data/sigs.npy")[1025]
    print(client.run(
    lambda qq: (lambda npmod=__import__('numpy'), wt=__import__('worker_tasks'):
                {
                    "has_local": False if wt.WORKER_LOCAL_DATA is None or wt.WORKER_LOCAL_DATA.size == 0 else True,
                    "local_shape": None if wt.WORKER_LOCAL_DATA is None else wt.WORKER_LOCAL_DATA.shape,
                    "dtype_match": None if wt.WORKER_LOCAL_DATA is None else (wt.WORKER_LOCAL_DATA.dtype == qq.dtype),
                    "exact_matches": None if wt.WORKER_LOCAL_DATA is None else int(npmod.any(npmod.all(wt.WORKER_LOCAL_DATA == qq, axis=1)))
                })(),
    q), flush=True)


# -----------------------------------------------------------
# POST /query endpoint
# -----------------------------------------------------------
@app.post('/query')
def query(req: QueryRequest):
    """
    Receives a query request containing a feature vector and an optional k value.
    Distributes the query to all Dask workers for local filtering, merges
    their responses, and returns the top-k candidates.

    Steps:
        1. Convert the input list to a NumPy array.
        2. If no precomputed edges exist, return an error.
        3. Submit local filtering tasks to all Dask workers.
        4. Gather partial results from all workers.
        5. Merge, sort, and truncate to top-k results.
    """
    # --- DEBUG: Print received JSON request ---
    print(f"[DEBUG] Received query: {req.json()}", flush=True)

    # Convert request vector to NumPy array
    q = np.asarray(req.vector, dtype=np.uint64)
    # --- DEBUG: Print first 10 elements and dtype ---
    print(f"[DEBUG] Query vector dtype: {q.dtype}", flush=True)
    print(f"[DEBUG] Query vector preview (first 10): {q[:10]}", flush=True)

    edges = GLOBAL_EDGES
    if edges is None:
        return {"error": "No edges precomputed"}

    # Submit query tasks to all active Dask workers
    workers = list(client.scheduler_info()['workers'].keys())
    # DEBUG: Output the workers for inspection
    print(f"[DEBUG] Active Dask workers: {workers}", flush=True)

    futures = []
    for wi, w in enumerate(workers):
        f = client.submit(
            lambda qq, ee, rank, total: __import__('worker_tasks').shard_qed_filter_local(
                qq, ee, rank, total, top_m=100
            ),
            q, edges, wi, len(workers),
            workers=[w]
        )
        futures.append(f)

    # Each candidate: ((shard_idx, row_idx), score)
    results = client.gather(futures)

    # Merge candidate lists from all workers
    merged = []
    for r in results:
        merged.extend(r)

    # Sort candidates by similarity score (descending)
    merged.sort(key=lambda x: x[1], reverse=True)
    topk = merged[:req.k]
    
    # NOTE: Further steps like resolving vector IDs or exact distance
    # Return JSON-serializable structure already (vector_preview is a list)
    return {"candidates": [{"id": cand[0], "score": cand[1], "vector_preview": cand[2]} for cand in topk]}