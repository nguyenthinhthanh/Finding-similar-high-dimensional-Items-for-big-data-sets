# benchmarks/benchmark_runner.py
import numpy as np
import time
import sys, os
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# Add current directory to Python import path (so local imports work)
sys.path.append(os.path.dirname(__file__))
from synth_data import make_synthetic
# ------------------------------------------------------------
# Benchmark framework for comparing similarity search methods
# (Brute-force, FAISS, LSH, etc.) on synthetic high-dimensional data.
# ------------------------------------------------------------

# ========== Metrics ==========
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


# ========== Search methods ==========
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


def lsh_search(queries, data, k=10, n_planes=10, n_tables=5):
    """
    Simple Locality Sensitive Hashing (LSH) baseline using random projections.
    - Each table projects data into low-dimensional hash codes.
    - Similar points are likely to share the same hash.
    """
    from sklearn.random_projection import GaussianRandomProjection
    n, d = data.shape
    # Build tables
    tables = [GaussianRandomProjection(n_components=n_planes) for _ in range(n_tables)]
    hashes = [np.sign(t.fit_transform(data)) for t in tables]

    all_results = []
    for q in queries:
        candidates = set()
        for t, h in zip(tables, hashes):
            hq = np.sign(t.transform(q.reshape(1, -1)))
            matches = np.where((h == hq).all(axis=1))[0]
            candidates.update(matches)
        if not candidates:
            candidates = np.random.choice(range(n), size=50, replace=False)
        cand_list = list(candidates)
        dists = np.linalg.norm(data[cand_list] - q, axis=1)
        topk = np.argsort(dists)[:k]
        all_results.append(np.array(cand_list)[topk])
    return np.array(all_results)


# ========== Benchmark runner ==========
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
                for rank, global_idx in enumerate(row):
                    shard_idx = global_idx // 5000
                    row_idx   = global_idx % 5000
                    vector_value = data[global_idx]
                    preview = vector_value[:10]
                    print(f"  Top-{rank+1}: global={global_idx:6d} (shard={shard_idx}, row={row_idx}) → preview={preview}")

    return pd.DataFrame(results)

SINGLE_TEST = 0
MERTRIC_TEST = 1
MODE = SINGLE_TEST

SHARD_SIZE = 5000

# ========== Main test ==========
if __name__ == "__main__":
    # Create synthetic data for test
    # make_synthetic(20000, 128, "data/raw.npy")
    data = np.load("data/raw.npy")

    # A specific 128-dimensional query vector
    query_vector = np.array([
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05,
        0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.11,
        0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 0.12, 0.23,
        0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 0.90, 0.91, 0.92, 0.93,
        0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.11, 0.12, 0.13, 0.14,
        0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24,
        0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34,
        0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44,
        0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54,
        0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64,
        0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74,
        0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84,
        0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92
    ])

    if MODE==SINGLE_TEST:
        queries = query_vector.reshape(1, -1)
    else:
        queries = data[:100]

    methods = {
        "Brute-force": brute_force_nn,
        "FAISS": faiss_search,
        "LSH": lsh_search,
        # "QED": qed_search,
    }

    df = run_benchmarks(data, queries, methods, k=5)
    print("\nBenchmark results:\n", df.to_string(index=False))
    df.to_csv("results_synthetic.csv", index=False)
 
