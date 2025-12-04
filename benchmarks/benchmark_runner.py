# benchmarks/benchmark_runner.py
import numpy as np
import time
import json
import sys, os
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# --- CẤU HÌNH ---
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "X.npy")         # Dữ liệu Vector mới
META_PATH = os.path.join(DATA_DIR, "meta.parquet")  # Từ điển mới
SHARD_SIZE = 100000                                 # Khớp với config index_builder

# ============================================================
# Utility: Generate curl command
# ============================================================
def save_curl_for_query(data, index, k=5, out_dir="benchmarks"):
    """Tạo file curl để test API với vector thực tế"""
    query_vector = data[index].tolist() # Float list
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "vector": query_vector,
        "k": k
    }
    
    # Lưu lệnh curl (dùng json.dumps để format đúng JSON cho float)
    json_body = json.dumps(payload)
    # Escape dấu ngoặc kép cho dòng lệnh bash
    json_body_escaped = json_body.replace('"', '\\"')
    
    curl_command = (
        f'curl -X POST "http://localhost:8000/query" '
        f'-H "Content-Type: application/json" '
        f'-d "{json_body_escaped}"'
    )

    out_path = os.path.join(out_dir, "curl_query.sh")
    with open(out_path, "w") as f:
        f.write(curl_command + "\n")
    print(f"Đã lưu lệnh test curl vào: {out_path}")

# ============================================================
# Metrics
# ============================================================
def recall_at_k(pred, truth, k):
    recalls = [len(set(p) & set(t)) / k for p, t in zip(pred, truth)]
    return np.mean(recalls)

def precision_at_k(pred, truth, k):
    precisions = [len(set(p) & set(t)) / len(set(p)) for p, t in zip(pred, truth)]
    return np.mean(precisions)

# ============================================================
# Search Methods
# ============================================================
def brute_force_nn(queries, data, k=10):
    """Tìm kiếm chính xác bằng vét cạn (Scan toàn bộ)"""
    # Tính khoảng cách Euclidean
    dists = euclidean_distances(queries, data)
    # Lấy top-k khoảng cách nhỏ nhất
    idx = np.argsort(dists, axis=1)[:, :k]
    return idx

def faiss_search(queries, data, k=10):
    """Tìm kiếm nhanh bằng thư viện FAISS (nếu có)"""
    try:
        import faiss
        # Sử dụng IndexFlatL2 (tương đương Brute-force nhưng tối ưu C++)
        index = faiss.IndexFlatL2(data.shape[1])
        index.add(data)
        _, I = index.search(queries, k)
        return I
    except ImportError:
        print("Chưa cài thư viện 'faiss-cpu'. Bỏ qua phương thức này.")
        return None

# ============================================================
# Main Runner
# ============================================================
def run_benchmarks(data, queries, methods, true_neighbors, k=10, docs_map=None):
    results = []
    
    for name, func in methods.items():
        print(f"\nRunning method: {name}...")
        start = time.time()
        
        try:
            pred_idx = func(queries, data, k)
        except Exception as e:
            print(f"Method {name} failed: {e}")
            continue
            
        if pred_idx is None: continue

        elapsed = time.time() - start
        latency = elapsed / len(queries)
        
        # Tính metrics
        rec = recall_at_k(pred_idx, true_neighbors, k)
        prec = precision_at_k(pred_idx, true_neighbors, k)
        
        results.append({
            "method": name,
            "recall@k": round(rec, 4),
            "precision@k": round(prec, 4),
            "latency_ms": round(latency * 1000, 3)
        })

        # --- In chi tiết kết quả của query đầu tiên để kiểm tra ---
        if name == "Brute-force": # Chỉ in chi tiết của thuật toán chuẩn
            print(f"\n--- Inspection for First Query ({name}) ---")
            q_idx = 0 # Xem query đầu tiên trong batch
            row = pred_idx[q_idx]
            
            print(f"Query Vector (first 5 dims): {queries[q_idx][:5]}")
            for rank, global_id in enumerate(row):
                # Mapping từ ID sang Word
                word = docs_map.get(global_id, "<Unknown>") if docs_map else "N/A"
                
                # Tính lại khoảng cách để verify
                dist = np.linalg.norm(queries[q_idx] - data[global_id])
                
                # Tính vị trí shard (giả lập logic server)
                shard_id = global_id // SHARD_SIZE
                row_id = global_id % SHARD_SIZE
                
                print(f"  Rank {rank+1}: ID={global_id:<6} (Shard={shard_id}, Row={row_id}) | Dist={dist:.4f} | Word='{word}'")

    return pd.DataFrame(results)

if __name__ == "__main__":
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("Lỗi: Không tìm thấy file X.npy. Hãy chạy synth_data.py trước.")
        sys.exit(1)
        
    data = np.load(DATA_PATH) # Float32
    print(f"Data loaded. Shape: {data.shape}, Dtype: {data.dtype}")

    # 2. Load Metadata (Dictionary)
    print(f"Loading metadata from {META_PATH}...")
    if os.path.exists(META_PATH):
        df = pd.read_parquet(META_PATH)
        # Tạo map: ID -> Word
        docs_map = dict(zip(df['id'], df['word']))
    else:
        print("Warning: Không tìm thấy meta.parquet, sẽ không hiển thị được từ ngữ.")
        docs_map = {}

    # 3. Setup Benchmark
    # Lấy 100 vector đầu tiên làm query test
    n_queries = 100
    queries = data[:n_queries]
    
    # Tạo curl command cho vector thứ 10 (ví dụ) để bạn test tay với server
    save_curl_for_query(data, index=10, k=5)

    # 4. Define Methods
    # Lưu ý: Chúng ta bỏ MinHashLSH vì nó không chạy trên Float32
    # Benchmark này chủ yếu so sánh độ chính xác của logic tìm kiếm
    methods = {
        "Brute-force": brute_force_nn, # Đây là Ground Truth (Chuẩn)
        "FAISS": faiss_search          # Thư viện tối ưu (nếu cài)
    }

    # 5. Run Ground Truth (để tính Recall/Precision)
    print("Computing Ground Truth (Brute-force)...")
    true_neighbors = brute_force_nn(queries, data, k=10)

    # 6. Run All
    df_results = run_benchmarks(data, queries, methods, true_neighbors, k=10, docs_map=docs_map)
    
    print("\n=== BENCHMARK RESULTS ===")
    print(df_results.to_string(index=False))