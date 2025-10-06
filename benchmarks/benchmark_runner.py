# benchmarks/benchmark_runner.py
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from benchmarks.synth_data import make_synthetic
import time

def brute_force_nn(query, data, k=10):
    dists = np.linalg.norm(data - query, axis=1)
    idx = np.argsort(dists)[:k]
    return idx, dists[idx]

if __name__ == '__main__':
    make_synthetic