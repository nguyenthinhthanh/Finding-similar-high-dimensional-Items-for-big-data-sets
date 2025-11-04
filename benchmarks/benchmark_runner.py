# benchmarks/benchmark_runner.py
import numpy as np
import time
import sys, os
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.random_projection import GaussianRandomProjection

# Add current directory to Python import path (so local imports work)
sys.path.append(os.path.dirname(__file__))
# from synth_data import make_synthetic
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
    # data = np.load("data/raw.npy")
    data = np.load("data/sigs.npy")

    # A specific 128-dimensional query vector
    # query_vector kiểu uint64
    query_vector = np.array([
        20671465220331927, 34175777397496750, 80829186156850025,
        88635534964285038, 9112867606025055, 1709117566375511,
        82917782605133416, 35790081631704231, 69571680940795994,
        9991748517474737, 13920169278362314, 63675045577996314,
        9853433044423600, 8035557461801026, 57762428873527925,
        10971748337416119, 30708069929342391, 35304531073323056,
        74681868667921862, 4598310885835408, 194068001033447611,
        44744794608484063, 128048678468233851, 61398546699259529,
        28118135734544055, 55287024053266251, 17564333542355796,
        30105059785170628, 13074715680636571, 12624839102519184,
        5909118797334551, 44067911108829631, 80901820057234369,
        29983841822035657, 3111033626678120, 56055167090094667,
        36918412124493904, 83390585036156095, 1667623468653129,
        126451123559935330, 13128189873249741, 23424605569357598,
        8019509209376302, 88347006485829618, 2797954171505209,
        11382256530431056, 66165878729366285, 2787148076279624,
        20795391085073308, 142699309776712426, 42413481982353202,
        21249846663489374, 135508195851363971, 261530593155492,
        65735867109004198, 10744475487415644, 41204202908876519,
        102043804233227067, 142517813587459710, 24695569700034265,
        108748127783445795, 101971308726405527, 38887565990448006,
        103498908244237462, 94695575630165705, 60240806105722837,
        9646307830059835, 19255578465757478, 34471767607596998,
        5685187937426200, 31829551987215286, 13616048838912635,
        25184831242953759, 14417366777521590, 37064059552713772,
        3033113895350854, 5191632450608742, 146341109503327012,
        156590079242801998, 17300687947001737, 34620875209026277,
        71373905262844640, 56189483757001070, 110876827368004702,
        96808376633596403, 42247040309219829, 35875807443554023,
        26130401147403247, 44176644943472856, 25626294473873061,
        20915035622628184, 184388894131164911, 7594062065944521,
        97853556634076165, 35563392508734417, 27539112482274769,
        22959186333280730, 21386137353824488, 24620354376094036,
        35920917529998096, 6624682608098277, 11219062568546792,
        72882798587813671, 14912027969644858, 27228740866559848,
        105403129441109219, 28266013502625795, 27398409827342045,
        27052297407082121, 52249695356106581, 8168775721601861,
        33573531459298906, 52966175500182171, 16788513626945106,
        27219837536740391, 75810378006175208, 74785443624135210,
        9509120959458506, 6683288823138342, 18333634903144794,
        22045121991861571, 43037955519778044, 4696816028734021,
        6951078643954329, 36618954585823496, 11225289160533895,
        47379742535648623, 55053311741533782
    ], dtype=np.uint64)

    assert query_vector.shape[0] == data.shape[1]
    queries = query_vector.reshape(1, -1)

    if MODE==SINGLE_TEST:
        print("Single Test....")
        queries = query_vector.reshape(1, -1)
    else:
        print("Metrics Test....")
        queries = data[:100]

    methods = {
        "Brute-force": brute_force_nn,
        "FAISS": faiss_search,
        "LSH": lsh_search
    }

    df = run_benchmarks(data, queries, methods, k=5)
    print("\nBenchmark results:\n", df.to_string(index=False))
    df.to_csv("results_synthetic.csv", index=False)
 
