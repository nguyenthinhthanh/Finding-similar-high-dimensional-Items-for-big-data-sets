# src/query_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from distributed import Client
import numpy as np
import sys, os
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
    vector: List[float]
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

@app.on_event("startup")    
def startup_event():
    """
    FastAPI lifecycle hook: called when the service starts.
    Ensures all Dask workers are reachable and have required modules loaded.
    """
    responses = client.run(lambda: "worker ready")
    print("[Startup] Connected workers:")
    for addr, msg in responses.items():
        print(f"  - {addr}: {msg}")

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

    # Convert request vector to NumPy array
    q = np.asarray(req.vector, dtype=float)
    edges = GLOBAL_EDGES
    if edges is None:
        return {"error": "No edges precomputed"}

    # Submit query tasks to all active Dask workers
    workers = list(client.scheduler_info()['workers'].keys())
    futures = []
    for w in workers:
        f = client.submit(lambda qq, ee: __import__('worker_tasks').shard_qed_filter_local(qq, ee, top_m=100), q, edges, workers=[w])
        futures.append(f)
    results = client.gather(futures)

    # Merge candidate lists from all workers
    merged = []
    for r in results:
        merged.extend(r)

    # Sort candidates by similarity score (descending)
    merged.sort(key=lambda x: x[1], reverse=True)
    topk = merged[:req.k]
    
    # NOTE: Further steps like resolving vector IDs or exact distance
    # computation are omitted for simplicity.
    return {"candidates": topk}