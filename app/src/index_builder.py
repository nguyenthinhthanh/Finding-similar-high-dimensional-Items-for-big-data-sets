# src/index_builder.py
import numpy as np
import sys, os
import argparse
from typing import Tuple

"""
Index Builder Script (SimHash-ready)
-----------------------------------
Splits a large (N, D) signature array into smaller shards (.npy).

Supports:
 - raw mode: save shards as (n_rows, D) uint8 (values 0/1)
 - packed mode: save shards as (n_rows, bytes_per_row) uint8 where each row = packed bits

Usage examples:
  # raw (default)
  python src/index_builder.py --data data/sigs.npy --out data/shards --shard-size 5000

  # packed bits (smaller disk, faster I/O)
  python src/index_builder.py --data data/sigs.npy --out data/shards --shard-size 5000 --pack-bits
"""

def ensure_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

def _validate_and_get_params(arr_shape: Tuple[int,int], pack_bits: bool):
    N, D = arr_shape
    if D != 128:
        print(f"[Warning] Signature dimension is {D}; code assumes 128 for SimHash but will proceed.")
    bytes_per_row = (D + 7) // 8 if pack_bits else None
    return N, D, bytes_per_row

def _save_shard_raw(shard: np.ndarray, out_path: str):
    """
    Save shard as (n_rows, D) uint8 array (0/1)
    """
    # ensure 0/1 uint8
    shard_u8 = shard.astype(np.uint8)
    np.save(out_path, shard_u8)

def _save_shard_packed(shard: np.ndarray, out_path: str):
    """
    Pack bits along axis 1 and save as (n_rows, bytes_per_row) uint8
    """
    # ensure 0/1 uint8
    shard_u8 = shard.astype(np.uint8)
    # pad to multiple of 8 automatically handled by packbits
    packed = np.packbits(shard_u8, axis=1)
    np.save(out_path, packed)

def split_and_save(data_path: str, out_dir: str, shard_size: int = 5000, pack_bits: bool = False):
    """
    Split a large dataset (N, D) into smaller .npy shards for easier processing.

    If pack_bits=True, each saved shard row is packed bytes (smaller on disk).
    """
    ensure_out_dir(out_dir)

    # Use memory-mapped load to avoid reading whole file into RAM
    print(f"[Info] Loading data (mmap) from {data_path} ...")
    arr = np.load(data_path, mmap_mode='r')
    N, D, bytes_per_row = _validate_and_get_params(arr.shape, pack_bits)

    i = 0
    for start in range(0, N, shard_size):
        end = min(N, start + shard_size)
        # slice from memmap; will not load entire arr into memory
        shard = arr[start:end]
        out_fname = os.path.join(out_dir, f"shard_{i:05d}.npy")
        if pack_bits:
            _save_shard_packed(shard, out_fname)
        else:
            _save_shard_raw(shard, out_fname)
        print(f"[Info] Wrote shard {i:05d}: rows {start}:{end} -> {out_fname}")
        i += 1

    print(f"[Info] Completed: wrote {i} shards to {out_dir}")

def print_hist_info_shard(shard: np.ndarray, name: str, packed: bool):
    """
    Print summary info for a shard. For packed shards we unpack first few rows for preview.
    """
    print(f"\n--- Info for {name} ---")
    print(f"Shape (on-disk): {shard.shape}, dtype={shard.dtype}")
    if packed:
        # unpack first rows for preview
        try:
            # load into memory small piece if necessary
            rows_to_preview = min(3, shard.shape[0])
            small = shard[:rows_to_preview]
            unpacked = np.unpackbits(small, axis=1)[:, :128]  # ensure 128 bits
            print(f"First {rows_to_preview} unpacked rows:\n{unpacked}")
            print(f"Min={unpacked.min()}, Max={unpacked.max()}")
        except Exception as e:
            print(f"[Warning] Failed to unpack preview: {e}")
    else:
        # raw 0/1 data
        rows_to_preview = min(3, shard.shape[0])
        small = shard[:rows_to_preview]
        print(f"First {rows_to_preview} rows:\n{small}")
        try:
            print(f"Min={small.min()}, Max={small.max()}")
        except:
            pass
    print("----------------------")

def print_all_shards_info(shard_dir: str, packed: bool):
    shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".npy")])
    if not shard_files:
        print(f"[Error] No shard files found in {shard_dir}")
        return
    for fname in shard_files:
        path = os.path.join(shard_dir, fname)
        shard = np.load(path, mmap_mode='r')
        print_hist_info_shard(shard, fname, packed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help="Path to input .npy (sigs.npy)")
    parser.add_argument('--out', required=True, help="Output directory for shards")
    parser.add_argument('--shard-size', type=int, default=5000)
    parser.add_argument('--inspect', action='store_true', help="Print info of all shards after splitting")
    parser.add_argument('--pack-bits', action='store_true', help="Pack bits per row to bytes (smaller files)")
    args = parser.parse_args()

    split_and_save(args.data, args.out, shard_size=args.shard_size, pack_bits=args.pack_bits)

    if args.inspect:
        print_all_shards_info(args.out, packed=args.pack_bits)
