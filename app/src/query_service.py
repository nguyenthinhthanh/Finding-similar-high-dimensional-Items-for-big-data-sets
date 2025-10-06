# src/query_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from distributed import Client
import numpy as np
import os
from typing import List, Tuple

app = FastAPI()
DASK_ADDR = os.environ.get('DASK_SCHEDULER_ADDRESS', 'tcp://scheduler:8786')
client = Client(DASK_ADDR)

class QueryRequest(BaseModel):
    vector: List[float]
    k: int = 10

# pre-load edges path
EDGES_PATH = os.environ.get('EDGES_PATH', '/data/hist_edges.npy')
if os.path.exists(EDGES_PATH):
    GLOBAL_EDGES = np.load(EDGES_PATH)
else:
    GLOBAL_EDGES = None

@app.on_event("startup")    
def startup_event():
    # ensure workers have modules available
    client.run(lambda: None)

@app.post('/query')
def query(req: QueryRequest):
    q = np.asarray(req.vector, dtype=float)
    edges = GLOBAL_EDGES
    if edges is None:
        return {"error": "No edges precomputed"}

    # submit tasks to all workers; target by worker address
    workers = list(client.scheduler_info()['workers'].keys())
    futures = []
    for w in workers:
        f = client.submit(lambda qq, ee: __import__('src.worker_tasks').query.worker_tasks.shard_qed_filter_local(qq, ee, top_m=100), q, edges, workers=[w])
        futures.append(f)
    results = client.gather(futures)

    # merge candidates
    merged = []
    for r in results:
        merged.extend(r)
    merged.sort(key=lambda x: x[1], reverse=True)
    topk = merged[:req.k]
    # compute exact distances (resolve ids -> vectors) omitted here for brevity
    return {"candidates": topk}