# src/index_builder.py
import numpy as np
import sys, os
import argparse
"""
Index Builder Script
--------------------
This script prepares data for distributed similarity search by:
1. Splitting a large (N, D) dataset into smaller shards (.npy files) for parallel processing.

Usage:
    python app/src/index_builder.py \
    --data data/sigs.npy \
    --out data/shards \
    --shard-size 5000 \
    --inspect
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
# Print shard for inspection / debugging
# ===============================================================
def print_hist_info(shard: np.ndarray, name: str):
    """
    Print summary information about a single shard for inspection.
    """
    print(f"\n--- Histogram Info for {name} ---")
    print(f"Shape: {shard.shape}")
    print(f"First 2 rows:\n{shard[:2]}")
    print(f"Min={shard.min():.4f}, Max={shard.max():.4f}")
    print("----------------------")

def print_all_shards_info(shard_dir: str):
    """
    Iterate through all .npy shard files in a directory and print info for each.
    """
    shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".npy")])
    if not shard_files:
        print(f"No shard files found in {shard_dir}")
        return

    for fname in shard_files:
        path = os.path.join(shard_dir, fname)
        shard = np.load(path)
        print_hist_info(shard, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--shard-size', type=int, default=100000)
    parser.add_argument('--inspect', action='store_true', help="Print info of all shards after splitting")
    args = parser.parse_args()

    split_and_save(args.data, args.out, shard_size=args.shard_size)

    if args.inspect:
        print_all_shards_info(args.out)