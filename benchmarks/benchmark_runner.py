# benchmarks/benchmark_runner.py
import numpy as np
import time
import json
import sys, os
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
from app.src.minhash_lsh import build_minhash_lsh_index, minhash_lsh_search
from benchmarks.synth_data import MinHash, shingle_document

# ============================================================
# benchmarks/benchmark_runner.py
# ============================================================
# Purpose:
#   This script benchmarks and compares multiple similarity search
#   algorithms (Brute-force, FAISS, LSH) on synthetic or real datasets.
#
#   It measures both performance (latency, throughput) and quality
#   metrics (Recall@k, Precision@k, MRR) for top-k nearest neighbor search.
#
# ============================================================

# ------------------------------------------------------------
# Load text documents and their corresponding IDs
# ------------------------------------------------------------
with open("data/docs.pkl", "rb") as f:
    docs = pickle.load(f)
with open("data/ids.pkl", "rb") as f:
    ids = pickle.load(f)

# ------------------------------------------------------------
# Utility: Generate a curl command for testing the REST API query
# ------------------------------------------------------------
def save_curl_for_query(data_path, index, k=5, out_dir="benchmarks"):
    """
    Create the file curl_query.sh to check a specific vector query.
    
    Parameters:
    data_path (str): path to the .npy file containing vector data
    index (int): index of the vector to check
    k (int): number of top-k results to retrieve
    out_dir (str): directory to save the curl_query.sh file
    """
    # Load data
    data = np.load(data_path)
    query_vector = data[index].tolist()

    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "vector": query_vector,
        "k": k
    }

    curl_command = (
        'curl -X POST "http://localhost:8000/query" '
        '-H "Content-Type: application/json" '
        f'-d "{json.dumps(payload).replace('"', '\\"')}"'
    )


    out_path = os.path.join(out_dir, "curl_query.sh")
    with open(out_path, "w") as f:
        f.write(curl_command + "\n")

# ------------------------------------------------------------
# Evaluation metrics for retrieval quality
# ------------------------------------------------------------
def recall_at_k(pred, truth, k):
    """
    Compute Recall@k:
      - Measures how many of the true nearest neighbors
        are correctly retrieved in the top-k predicted list.
    """
    recalls = [len(set(p) & set(t)) / k for p, t in zip(pred, truth)]
    return np.mean(recalls)

def precision_at_k(pred, truth, k):
    """
    Compute Precision@k:
      - Measures how many of the retrieved neighbors
        are actually correct among all top-k predictions.
    """
    precisions = [len(set(p) & set(t)) / len(set(p)) for p, t in zip(pred, truth)]
    return np.mean(precisions)

def mean_reciprocal_rank(pred, truth):
    """
    Compute Mean Reciprocal Rank (MRR):
      - Measures ranking quality by averaging the reciprocal
        of the rank position of the first correct neighbor.
    """
    ranks = []
    for p, t in zip(pred, truth):
        rank = 0
        for i, val in enumerate(p):
            if val in t:
                rank = 1.0 / (i + 1)
                break
        ranks.append(rank)
    return np.mean(ranks)


# ------------------------------------------------------------
# Similarity search methods
# ------------------------------------------------------------
def brute_force_nn(queries, data, k=10):
    """
    Brute-force nearest neighbor search (exact baseline).
    - Compute full distance matrix between queries and all data points.
    - Sort each row to get top-k smallest distances.
    """
    dists = euclidean_distances(queries, data)
    idx = np.argsort(dists, axis=1)[:, :k]
    return idx

def faiss_search(queries, data, k=10):
    """
    Approximate nearest neighbor search using Facebook AI's FAISS library.
    - Uses a flat (non-quantized) L2 index for simplicity.
    """
    import faiss
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(data)
    _, I = index.search(queries, k)
    return I

# ------------------------------------------------------------
# Benchmark runner: executes all methods and compares performance
# ------------------------------------------------------------
def run_benchmarks(data, queries, methods, k=10):
    print(f"Running {len(methods)} methods on data {data.shape}, queries={queries.shape[0]}...")
    truth = brute_force_nn(queries, data, k)
    results = []

    for name, func in methods.items():
        print(f"\nRunning {name}...")
        start = time.time()
        idx = func(queries, data, k)
        elapsed = time.time() - start
        latency = elapsed / len(queries)
        throughput = 1.0 / latency

        recall = recall_at_k(idx, truth, k)
        precision = precision_at_k(idx, truth, k)
        mrr = mean_reciprocal_rank(idx, truth)

        results.append({
            "method": name,
            "recall@k": round(recall, 4),
            "precision@k": round(precision, 4),
            "MRR": round(mrr, 4),
            "latency_ms": round(latency * 1000, 3),
            "throughput_qps": round(throughput, 2),
        })

        if MODE == SINGLE_TEST:
            for qi, row in enumerate(idx):
                print(f"\nQuery {qi}:")

                query_vec = queries[qi].ravel()
                for rank, global_idx in enumerate(row):
                    if int(global_idx) == -1:
                        print(f"  Top-{rank+1}: <padded -1> (no result)")
                        continue

                    shard_idx = int(global_idx) // SHARD_SIZE
                    row_idx   = int(global_idx) % SHARD_SIZE
                    vector_value = data[int(global_idx)]
                    preview = vector_value[:10]

                    doc_text = docs[int(global_idx)]
                    doc_id = ids[int(global_idx)]

                    # Euclidean distance (L2) between query and candidate
                    # ensure cast to float for safe math if dtype is uint64
                    try:
                        # if signatures are integer types, convert to float for distance
                        dist = float(np.linalg.norm(query_vec.astype(float) - vector_value.astype(float)))
                    except Exception:
                        # Fallback: use sklearn pairwise for safety
                        from sklearn.metrics.pairwise import euclidean_distances
                        dist = float(euclidean_distances(query_vec.reshape(1, -1).astype(float),
                                                        vector_value.reshape(1, -1).astype(float))[0, 0])

                    # If using MinHash signatures, also report estimated Jaccard similarity:
                    # fraction of positions in signature that are equal (MinHash property).
                    try:
                        jacc_est = float(np.count_nonzero(query_vec == vector_value) / query_vec.shape[0])
                    except Exception:
                        jacc_est = None

                    print(f"  Top-{rank+1}: global={global_idx:6d} (shard={shard_idx}, row={row_idx}): preview={preview}")
                    print(f"        preview={preview}")
                    print(f"        doc_id={doc_id} -> {doc_text[:100]}...")
                    print(f"        Euclidean L2 distance = {dist:.6f}", end="")
                    if jacc_est is not None:
                        print(f"  |  est.Jaccard(from sig) = {jacc_est:.4f}")
                    else:
                        print("")

    return pd.DataFrame(results)

SINGLE_TEST = 0
MERTRIC_TEST = 1
MODE = SINGLE_TEST

SHARD_SIZE = 5000

# ========== Main test ==========
if __name__ == "__main__":
    # Load MinHash signatures (precomputed features)
    data = np.load("data/sigs.npy")

    # Pick one specific query vector for inspection
    query_vector = data[1025].copy()
    print("Query MinHash signature shape:", query_vector.shape)
    # print("Query MinHash signature (sample 10):", query_vector[:10])

    # Save command for test
    save_curl_for_query("data/sigs.npy", index=1025, k=5)

    # Build lsh banding
    lsh_index = build_minhash_lsh_index(data=data)

    # Wrappers for methods to match expected function signature (queries, data, k)
    def lsh_wrapper(queries_arr, data_arr, k=10):
        return minhash_lsh_search(queries_arr, data_arr, k=k, lsh_index=lsh_index)

    if MODE==SINGLE_TEST:
        print("Single Test....")
        assert query_vector.shape[0] == data.shape[1]
        queries = query_vector.reshape(1, -1)
    else:
        print("Metrics Test....")
        queries = data[:100]

    methods = {
        "Brute-force": brute_force_nn,
        "FAISS": faiss_search,
        "LSH": lsh_wrapper
    }

    df = run_benchmarks(data, queries, methods, k=5)
    print("\nBenchmark results:\n", df.to_string(index=False))
    df.to_csv("results_synthetic.csv", index=False)
 
