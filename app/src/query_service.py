# src/query_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from distributed import Client
import numpy as np
import sys, os
import time
from typing import List

# Add current directory to Python import path
sys.path.append(os.path.dirname(__file__))

# -----------------------------------------------------------
# Dask-based Query Service (CLEANED & FIXED)
# -----------------------------------------------------------
app = FastAPI()

DASK_ADDR = os.environ.get('DASK_SCHEDULER_ADDRESS', 'tcp://scheduler:8786')
client = Client(DASK_ADDR)

class QueryRequest(BaseModel):
    vector: List[float] # Đã sửa thành float cho đúng bản chất
    k: int = 10

# -----------------------------------------------------------
# Edge Handling (Legacy support for qed.py signature)
# -----------------------------------------------------------
# Vì hàm quantify_score trong qed.py vẫn nhận tham số 'edges',
# ta tạo một biến dummy để truyền vào, dù không dùng tới.
GLOBAL_EDGES = np.array([0]) # Dummy array

# ------------------------------------------------------------------
# Helper: wait for workers
# ------------------------------------------------------------------
def wait_for_workers(client, timeout=30, expected_count=1): # Giảm expected xuống 1 để dễ debug
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            sinfo = client.scheduler_info()
            workers = sinfo.get("workers", {})
            if len(workers) >= expected_count:
                return workers
        except Exception as e:
            print(f"[Error] {e}", flush=True)
        time.sleep(1.0)
    print(f"[Warning] Timeout waiting for workers. Current count: {len(workers) if 'workers' in locals() else 0}", flush=True)
    return workers if 'workers' in locals() else {}

# ------------------------------------------------------------------
# Startup Event
# ------------------------------------------------------------------
@app.on_event("startup")    
def startup_event():
    print("[Startup] Connecting to Dask Scheduler...", flush=True)
    
    # 1. Wait for workers
    workers = wait_for_workers(client, timeout=30, expected_count=1)
    
    if not workers:
        print("[Startup] WARNING: No workers found! Queries will fail.", flush=True)
        return

    print(f"[Startup] Found {len(workers)} workers: {list(workers.keys())}", flush=True)

    # 2. Trigger Data Loading on Workers
    # (Gọi hàm build_local_lsh_init nhưng thực chất là để load X.npy vào RAM)
    print("[Startup] Commanding workers to load shards (Float32)...", flush=True)
    
    # Config dummy (không quan trọng vì ta đã tắt LSH)
    BANDS = 32
    MAX_BUCKET = 5000
    
    import worker_tasks
    
    for wi, addr in enumerate(list(workers.keys())):
        try:
            client.run(worker_tasks.build_local_lsh_init,
                        wi, len(workers), BANDS, MAX_BUCKET,
                        workers=[addr])
            print(f"[Startup] Worker {addr} (rank {wi}) initialized.", flush=True)
        except Exception as e:
            print(f"[Startup] Failed to init worker {addr}: {e}", flush=True)

    print("[Startup] System Ready.", flush=True)


# -----------------------------------------------------------
# POST /query endpoint
# -----------------------------------------------------------
@app.post('/query')
def query(req: QueryRequest):
    # 1. Convert Input
    # QUAN TRỌNG: Chuyển về float32 để khớp với dữ liệu X.npy
    q = np.asarray(req.vector, dtype=np.float32)

    # 2. Get Workers
    workers = list(client.scheduler_info()['workers'].keys())
    if not workers:
        return {"error": "No workers available"}

    # 3. Distributed Query (Map Phase)
    futures = []
    for wi, w in enumerate(workers):
        # Gọi hàm filter trên từng worker
        f = client.submit(
            lambda qq, ee, rank, total: __import__('worker_tasks').shard_qed_filter_local(
                qq, ee, rank, total, top_m=req.k * 2 # Lấy dư ra một chút ở mỗi worker
            ),
            q, GLOBAL_EDGES, wi, len(workers),
            workers=[w]
        )
        futures.append(f)

    # 4. Gather Results (Reduce Phase)
    try:
        results = client.gather(futures) # List of lists
    except Exception as e:
        return {"error": f"Dask computation failed: {str(e)}"}

    # 5. Merge & Sort
    merged = []
    for r in results:
        merged.extend(r)

    # Sort theo score giảm dần (Score càng cao càng giống)
    # Lưu ý: qed.py đang trả về Negative Distance (vd: -0.5, -1.2).
    # -0.5 lớn hơn -1.2 -> Sort Reverse=True là ĐÚNG.
    merged.sort(key=lambda x: x[1], reverse=True)
    
    topk = merged[:req.k]
    
    # 6. Format Output
    return {
        "candidates": [
            {
                "id": cand[0],          # (shard_idx, row_idx)
                "score": cand[1],       # Negative L2 Distance
                "vector_preview": cand[2]
            } for cand in topk
        ]
    }