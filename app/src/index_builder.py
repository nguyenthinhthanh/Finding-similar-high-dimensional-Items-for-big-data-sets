# src/index_builder.py
import numpy as np
import sys, os
import argparse

# Add current directory to Python import path (so local imports work)
sys.path.append(os.path.dirname(__file__))
from qed import build_histograms
"""
Index Builder Script
--------------------
This script prepares data for distributed similarity search by:
1. Splitting a large (N, D) dataset into smaller shards (.npy files) for parallel processing.
2. Sampling data from shards to build a global histogram (used for quantization binning).

Usage:
    python src/index_builder.py \
        --data data/raw.npy \
        --out data/shards \
        --edges-out data/hist_edges.npy \
        --shard-size 5000
"""

# ===============================================================
# Split the dataset into shards
# ===============================================================
def split_and_save(data_path: str, out_dir: str, shard_size: int = 100000):
    """
    Split a large dataset (N, D) into smaller .npy shards for easier processing.
    Each shard contains up to `shard_size` rows.
    """
    os.makedirs(out_dir, exist_ok=True)
    arr = np.load(data_path) # expect (N, D)
    N = arr.shape[0]
    i = 0
    for start in range(0, N, shard_size):
        end = min(N, start + shard_size)
        shard = arr[start:end]
        np.save(os.path.join(out_dir, f"shard_{i}.npy"), shard)
        i += 1
    print(f"Wrote {i} shards to {out_dir}")

# ===============================================================
# Print histogram metadata (for inspection / debugging)
# ===============================================================
def print_hist_info(edges: np.ndarray, counts: np.ndarray):
    """
    Print summary information about histogram edges and counts for inspection.
    """
    print("\n--- Histogram Info ---")
    print(f"Edges shape: {edges.shape}")
    print(f"Counts shape: {counts.shape}")
    print(f"Edges (first 5 rows):\n{edges[:5]}")
    print(f"Counts (first 5 rows):\n{counts[:5]}")
    print(f"Edges min={edges.min():.4f}, max={edges.max():.4f}")
    print(f"Counts min={counts.min()}, max={counts.max()}")
    print("----------------------\n")

# ===============================================================
# Build a global histogram (quantization edges)
# ===============================================================
def build_global_hist(edges_out: str, shards_dir: str, n_bins: int = 256):
    """
    Build global histogram edges from sampled shards.
    
    Parameters:
        edges_out (str): Path to save the resulting histogram edges (.npy file).
        shards_dir (str): Directory containing data shards (.npy files).
        n_bins (int): Number of histogram bins per feature dimension.
    
    Notes:
        - Randomly collects samples (up to 200,000 rows total) from shards.
        - Uses these samples to estimate global bin edges for quantization.
        - These edges will later be used by distributed workers for vector quantization.
    """
    # Naive: sample from shards to build global hist
    sample = []
    for fn in os.listdir(shards_dir):
        if fn.endswith('.npy'):
            arr = np.load(os.path.join(shards_dir, fn))
            sample.append(arr)
            if sum(x.shape[0] for x in sample) >= 200000:
                raise ValueError("Error: Total sample size exceeded 200000")
                break
    sample = np.vstack(sample)
    N, D = sample.shape
    edges, counts = build_histograms(sample, n_bins=n_bins)
    np.save(edges_out, edges)
    print(f"Saved edges to {edges_out}")

    # Print info for debugging / verification
    # print_hist_info(edges, counts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--shard-size', type=int, default=100000)
    parser.add_argument('--edges-out', default='/data/hist_edges.npy')
    args = parser.parse_args()
    split_and_save(args.data, args.out, shard_size=args.shard_size)
    build_global_hist(args.edges_out, args.out)