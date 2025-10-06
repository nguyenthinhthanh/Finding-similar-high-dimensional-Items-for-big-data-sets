# src/index_builder.py
import numpy as np
import os
import argparse
from src.qed import build_histograms


def split_and_save(data_path: str, out_dir: str, shard_size: int = 100000):
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

def build_global_hist(edges_out: str, shards_dir: str, n_bins: int = 256):
    # naive: sample from shards to build global hist
    sample = []
    for fn in os.listdir(shards_dir):
        if fn.endswith('.npy'):
            arr = np.load(os.path.join(shards_dir, fn))
            sample.append(arr)
            if sum(x.shape[0] for x in sample) >= 200000:
                break
    sample = np.vstack(sample)
    edges, counts = build_histograms(sample, n_bins=n_bins)
    np.save(edges_out, edges)
    print(f"Saved edges to {edges_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--shard-size', type=int, default=100000)
    parser.add_argument('--edges-out', default='/data/hist_edges.npy')
    args = parser.parse_args()
    split_and_save(args.data, args.out, shard_size=args.shard_size)
    build_global_hist(args.edges_out, args.out)