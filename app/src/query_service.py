# src/query_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from distributed import Client
import numpy as np
import sys, os
import pandas as pd
import pickle
from typing import List

# --- CẤU HÌNH ---
sys.path.append(os.path.dirname(__file__))

app = FastAPI()
DASK_ADDR = os.environ.get('DASK_SCHEDULER_ADDRESS', 'tcp://scheduler:8786')
client = Client(DASK_ADDR)

# Các đường dẫn file cấu hình
META_PATH = os.environ.get('META_PATH', 'data/meta.parquet')
MINHASH_META_PATH = os.environ.get('MINHASH_META', 'data/minhash_meta.pkl')
SEMANTIC_CONFIG_PATH = 'data/semantic_config.pkl' # File quan trọng cho Semantic
SHARD_SIZE = 5000 

# Biến toàn cục (Global State)
GLOBAL_META_DF = None
GLOBAL_SEMANTIC_DATA = None # Chứa projections và vocab (cho Semantic)
GLOBAL_MINHASH_CONFIG = None # Chứa config băm (cho MinHash)

# Model nhận request
class TextQueryRequest(BaseModel):
    text: str
    k: int = 10

def wait_for_workers(client, timeout=120): # Tăng time out lên 120s
    import time
    print("[Startup] Đang quét tìm Worker...", flush=True)
    start = time.time()
    
    while time.time() - start < timeout:
        # Lấy danh sách worker từ Scheduler
        workers = client.scheduler_info().get("workers", {})
        
        if len(workers) > 0:
            print(f"[Startup] ✅ Đã tìm thấy {len(workers)} Worker(s) đang online!", flush=True)
            # Chờ thêm 5s để Worker ổn định hẳn
            time.sleep(5) 
            return workers
            
        print(f"[Startup] ⏳ Chưa thấy Worker nào... Đợi tiếp... ({int(time.time() - start)}s)", flush=True)
        time.sleep(2) # Ngủ 2s rồi check lại
        
    print("[CRITICAL] ❌ Hết giờ! Không tìm thấy Worker nào cả. Kiểm tra lại cửa sổ Worker đi!", flush=True)
    return {}

@app.on_event("startup")    
def startup_event():
    global GLOBAL_META_DF, GLOBAL_SEMANTIC_DATA, GLOBAL_MINHASH_CONFIG
    
    print("[Startup] Connecting to Dask...", flush=True)
    workers = wait_for_workers(client)
    
    # 1. Trigger Worker LSH Build (Logic này giống nhau cho cả 2 chế độ)
    import worker_tasks
    BANDS = 32; MAX_BUCKET = 5000
    for wi, addr in enumerate(list(workers.keys())):
        client.run(worker_tasks.build_local_lsh_init, wi, len(workers), BANDS, MAX_BUCKET, workers=[addr])

    # 2. Load Metadata (ID -> Word) để trả kết quả
    if os.path.exists(META_PATH):
        try:
            GLOBAL_META_DF = pd.read_parquet(META_PATH)
            print(f"[Startup] Loaded Meta: {len(GLOBAL_META_DF)} words")
        except:
            print("[Warning] Lỗi đọc file Meta parquet")

    # 3. TỰ ĐỘNG PHÁT HIỆN CHẾ ĐỘ (QUAN TRỌNG)
    if os.path.exists(SEMANTIC_CONFIG_PATH):
        print("[Startup] ---> PHÁT HIỆN CHẾ ĐỘ: SEMANTIC (NGỮ NGHĨA) <---")
        with open(SEMANTIC_CONFIG_PATH, "rb") as f:
            GLOBAL_SEMANTIC_DATA = pickle.load(f)
            print("[Startup] Đã load Semantic Config (Projections & Mini Vocab)")
            
    elif os.path.exists(MINHASH_META_PATH):
        print("[Startup] ---> PHÁT HIỆN CHẾ ĐỘ: MINHASH (MẶT CHỮ) <---")
        with open(MINHASH_META_PATH, "rb") as f:
            GLOBAL_MINHASH_CONFIG = pickle.load(f)

@app.post('/search_text')
def search_text(req: TextQueryRequest):
    query_vector = None
    
    # === NHÁNH 1: XỬ LÝ THEO NGỮ NGHĨA (SEMANTIC) ===
    if GLOBAL_SEMANTIC_DATA is not None:
        projections = GLOBAL_SEMANTIC_DATA["projections"]
        vocab = GLOBAL_SEMANTIC_DATA["mini_vocab"]
        
        # Bước 1: Tra từ trong từ điển mini
        # (Vì ta không load hết 1.5 triệu vector lên RAM Service được, chỉ load top 50k)
        word = req.text
        if word not in vocab:
            return {
                "query": word, 
                "results": [], 
                "error": f"Từ '{word}' không có trong từ điển phổ biến (Top 50k). Hãy thử 'king', 'computer', 'school'..."
            }
        
        # Bước 2: Lấy vector 300 chiều
        vec_300d = vocab[word]
        
        # Bước 3: SimHash (Chiếu vector -> Signature)
        # Dot product: (300,) dot (300, 128) -> (128,)
        dots = np.dot(vec_300d, projections)
        query_vector = (dots > 0).astype(np.uint64)

    # === NHÁNH 2: XỬ LÝ THEO MẶT CHỮ (MINHASH CŨ) ===
    elif GLOBAL_MINHASH_CONFIG is not None:
        # Import lại logic tạo shingle ở đây để tránh phụ thuộc file ngoài
        import hashlib
        _PRIME = (1 << 61) - 1
        def _hash(s): return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:8], "big") % _PRIME
        
        # Cấu hình MinHash
        num_perm = GLOBAL_MINHASH_CONFIG["num_perm"]
        k = GLOBAL_MINHASH_CONFIG.get("k_shingle", 3)
        
        # Tạo Shingle
        if len(req.text) < k: shingles = {req.text}
        else: shingles = {req.text[i:i+k] for i in range(len(req.text)-k+1)}
        
        # Tạo Signature
        mh_a = np.random.RandomState(42).randint(1, _PRIME, size=num_perm, dtype=np.int64)
        mh_b = np.random.RandomState(42).randint(0, _PRIME, size=num_perm, dtype=np.int64)
        
        sh_ints = np.array([_hash(s) for s in shingles], dtype=np.int64)
        if len(sh_ints) == 0: query_vector = np.full(num_perm, _PRIME, dtype=np.uint64)
        else:
            vals = (mh_a[:, None] * sh_ints + mh_b[:, None]) % _PRIME
            query_vector = np.min(vals, axis=1).astype(np.uint64)

    # === GỬI ĐI TÌM KIẾM (CHUNG CHO CẢ 2) ===
    if query_vector is None:
        return {"error": "Hệ thống chưa sẵn sàng hoặc không xác định được chế độ."}

    # Gửi vector xuống các Worker
    workers = list(client.scheduler_info()['workers'].keys())
    futures = []
    for wi, w in enumerate(workers):
        f = client.submit(
            lambda qq, rank, total: __import__('worker_tasks').shard_qed_filter_local(qq, rank, total, top_m=100),
            query_vector, wi, len(workers), workers=[w]
        )
        futures.append(f)
    
    # Gom kết quả
    results = client.gather(futures)
    merged = []
    for r in results: merged.extend(r)
    merged.sort(key=lambda x: x[1], reverse=True)
    
    # Giải mã ID -> Word
    decoded = []
    for cand in merged[:req.k]:
        shard_idx, row_idx = cand[0]
        global_idx = shard_idx * SHARD_SIZE + row_idx # SHARD_SIZE phải khớp index_builder
        
        w_text = f"ID_{global_idx}"
        if GLOBAL_META_DF is not None and 0 <= global_idx < len(GLOBAL_META_DF):
            w_text = GLOBAL_META_DF.iloc[global_idx]["word"]
        
        decoded.append({"word": w_text, "score": cand[1]})

    return {"query": req.text, "results": decoded}