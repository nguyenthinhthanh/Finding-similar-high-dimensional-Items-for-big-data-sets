# benchmarks/benchmark_runner.py
import numpy as np
import time
import json
import sys, os
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

# ============================================================
# benchmark_runner.py (Adapted for SimHash signatures)
# - Uses Hamming distance for exact comparison of SimHash bit signatures.
# - Provides a simple banding-based LSH index for SimHash (band_size * bands = 128).
# - Keeps FAISS option (converts bits to float) for comparison.
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
    For SimHash we write the bit-array as a list of 0/1.
    """
    data = np.load(data_path)
    query_vector = data[index].tolist()

    os.makedirs(out_dir, exist_ok=True)

    payload = {
        "signature": query_vector,
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
# (unchanged)
# ------------------------------------------------------------
def recall_at_k(pred, truth, k):
    recalls = [len(set(p) & set(t)) / k for p, t in zip(pred, truth)]
    return np.mean(recalls)

def precision_at_k(pred, truth, k):
    precisions = [len(set(p) & set(t)) / len(set(p)) if len(set(p))>0 else 0.0 for p, t in zip(pred, truth)]
    return np.mean(precisions)

def mean_reciprocal_rank(pred, truth):
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
# Hamming / utility functions for SimHash (signatures are arrays of 0/1)
# ------------------------------------------------------------
def hamming_distances_single(query, data):
    """
    Compute Hamming distance from single query (1D array of 0/1) to all rows in data.
    Returns an (N,) int array.
    """
    # Convert to boolean for fast XOR and sum
    q_bool = query.astype(bool)
    d_bool = data.astype(bool)
    # XOR then sum along axis
    return np.count_nonzero(np.logical_xor(d_bool, q_bool), axis=1)

def brute_force_hamming_nn(queries, data, k=10):
    """
    Brute-force using Hamming distance for SimHash bit signatures.
    queries: (Q, D) 0/1
    data: (N, D) 0/1
    returns: (Q, k) indices (int)
    """
    Q = queries.shape[0]
    N = data.shape[0]
    result = np.full((Q, k), -1, dtype=int)
    for i in range(Q):
        dists = hamming_distances_single(queries[i], data)
        idx = np.argsort(dists)[:k]
        # If N < k, pad with -1 already
        result[i, :len(idx)] = idx
    return result

# ------------------------------------------------------------
# FAISS wrapper (keeps existing behavior, convert bits->float)
# ------------------------------------------------------------
def faiss_search(queries, data, k=10):
    """
    Use FAISS with L2 on float-converted signatures.
    Note: This is approximate for bit-signatures but useful for performance comparison.
    """
    import faiss
    # convert to float32
    data_f = data.astype(np.float32)
    queries_f = queries.astype(np.float32)
    index = faiss.IndexFlatL2(data_f.shape[1])
    index.add(data_f)
    _, I = index.search(queries_f, k)
    return I

# ------------------------------------------------------------
# Simple banding-based LSH for SimHash
# ------------------------------------------------------------
def build_simhash_lsh_index(data, bands=8, band_size=16):
    """
    Build an inverted index mapping (band_idx, band_key) -> list(indices).
    data: (N, 128) array of 0/1
    bands * band_size must == 128
    """
    assert bands * band_size == data.shape[1], "bands * band_size must equal signature length"
    index = defaultdict(list)
    N = data.shape[0]
    for i in range(N):
        row = data[i]
        for b in range(bands):
            start = b * band_size
            end = start + band_size
            band = row[start:end].astype(np.uint8)
            # Pack bits into minimal bytes, then use bytes as key
            key = bytes(np.packbits(band))
            index[(b, key)].append(i)
    # Convert defaultdict to normal dict for pickling/inspection
    return {
        "bands": bands,
        "band_size": band_size,
        "inv_index": dict(index)
    }

def simhash_lsh_search(queries, data, k=10, lsh_index=None, max_candidates=2000):
    """
    For each query:
      - collect candidate ids from matching band buckets
      - compute exact Hamming distance to candidates
      - return top-k indices (pad with -1)
    If candidate set is empty or too small, fallback to brute-force Hamming.
    """
    bands = lsh_index["bands"]
    band_size = lsh_index["band_size"]
    inv_index = lsh_index["inv_index"]
    Q = queries.shape[0]
    result = np.full((Q, k), -1, dtype=int)

    for qi in range(Q):
        q = queries[qi]
        candidates = set()
        for b in range(bands):
            start = b * band_size
            end = start + band_size
            band = q[start:end].astype(np.uint8)
            key = bytes(np.packbits(band))
            bucket = inv_index.get((b, key))
            if bucket:
                candidates.update(bucket)
            # small optimization: early stop if many candidates
            if len(candidates) >= max_candidates:
                break

        if not candidates:
            # fallback to full brute-force
            dists = hamming_distances_single(q, data)
            idx = np.argsort(dists)[:k]
            result[qi, :len(idx)] = idx
            continue

        cand_arr = np.fromiter(candidates, dtype=int)
        cand_vecs = data[cand_arr]
        dists = np.count_nonzero(np.logical_xor(cand_vecs.astype(bool), q.astype(bool)), axis=1)
        order = np.argsort(dists)
        topk = cand_arr[order][:k]
        result[qi, :len(topk)] = topk

        # If not enough candidates to fill k, we can fill remaining with nearest from full set
        if len(topk) < k:
            # compute on full set for the remaining
            full_dists = hamming_distances_single(q, data)
            global_order = np.argsort(full_dists)
            # pick items from global_order that are not already in topk
            fill = []
            for idx in global_order:
                if idx in topk:
                    continue
                fill.append(idx)
                if len(fill) + len(topk) >= k:
                    break
            if fill:
                start_fill = len(topk)
                result[qi, start_fill:start_fill+len(fill)] = fill

    return result

# ------------------------------------------------------------
# Benchmark runner: executes all methods and compares performance
# ------------------------------------------------------------
def run_benchmarks(data, queries, methods, k=10):
    print(f"Running {len(methods)} methods on data {data.shape}, queries={queries.shape[0]}...")
    # truth: brute-force hamming (exact)
    truth = brute_force_hamming_nn(queries, data, k)
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

        # Single-test verbosity (prints detailed neighbors)
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
                    preview = vector_value[:40]  # preview bits

                    doc_text = docs[int(global_idx)]
                    doc_id = ids[int(global_idx)]

                    # Hamming distance
                    try:
                        dist = int(np.count_nonzero(np.logical_xor(query_vec.astype(bool), vector_value.astype(bool))))
                    except Exception:
                        dist = None

                    # estimate bit-equality fraction as similarity
                    try:
                        sim_est = float(np.count_nonzero(query_vec == vector_value) / query_vec.shape[0])
                    except Exception:
                        sim_est = None

                    print(f"  Top-{rank+1}: global={global_idx:6d} (shard={shard_idx}, row={row_idx}): preview={preview}")
                    print(f"        doc_id={doc_id} -> {doc_text[:100]}...")
                    if dist is not None:
                        print(f"        Hamming distance = {dist}", end="")
                    if sim_est is not None:
                        print(f"  |  est.bit_similarity = {sim_est:.4f}")
                    else:
                        print("")

    return pd.DataFrame(results)

SINGLE_TEST = 0
MERTRIC_TEST = 1
MODE = SINGLE_TEST

SHARD_SIZE = 5000

# ========== Main test ==========
if __name__ == "__main__":
    # Load SimHash signatures (precomputed features)
    # Expect shape: (N, 128) with values 0 or 1 (dtype uint64 or uint8)
    data = np.load("data/sigs.npy")
    # if dtype is uint64 but each element is 0/1 it's fine; cast to uint8 for quicker ops
    data = data.astype(np.uint8)

    # Pick one specific query vector for inspection
    query_text = docs[int(1025)]
    query_vector = data[1025].copy()
    print(f"Query SimHash signature shape: {query_vector.shape} text: {query_text[:100]}...")
    print("-----> Query SimHash signature (sample 10):", query_vector[:10])

    # Save command for test
    save_curl_for_query("data/sigs.npy", index=1025, k=5)

    # Build lsh banding for SimHash
    lsh_index = build_simhash_lsh_index(data=data, bands=8, band_size=16)

    # Wrappers for methods to match expected function signature (queries, data, k)
    def lsh_wrapper(queries_arr, data_arr, k=10):
        return simhash_lsh_search(queries_arr, data_arr, k=k, lsh_index=lsh_index)

    if MODE==SINGLE_TEST:
        print("Single Test....")
        assert query_vector.shape[0] == data.shape[1]
        queries = query_vector.reshape(1, -1)
    else:
        print("Metrics Test....")
        queries = data[:100]

    methods = {
        "Brute-force(Hamming)": brute_force_hamming_nn,
        "FAISS(L2-on-floats)": faiss_search,
        "LSH(SimHash-band)": lsh_wrapper
    }

    df = run_benchmarks(data, queries, methods, k=5)
    print("\nBenchmark results:\n", df.to_string(index=False))
    df.to_csv("results_simhash.csv", index=False)
