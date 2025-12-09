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
    python src/index_builder.py \
    --data data/X.npy \
    --out data/shards \
    --shard-size 100000 \
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
    
    print(f"[INDEX BUILDER] Info: Loading data from {data_path} (mmap mode)...")
    # TỐI ƯU: Sử dụng mmap_mode='r' để không load toàn bộ file vào RAM
    # Giúp cắt file 10GB hay 100GB vẫn chạy tốt trên laptop RAM 4GB
    arr = np.load(data_path, mmap_mode='r') 
    
    N = arr.shape[0]
    print(f"[INDEX BUILDER] Info: Original Data Shape: {arr.shape}, Dtype: {arr.dtype}")
    
    i = 0
    for start in range(0, N, shard_size):
        end = min(N, start + shard_size)
        # Khi slice ở đây, numpy mới thực sự đọc dữ liệu từ ổ cứng
        shard = arr[start:end] 
        
        # Lưu file con
        out_path = os.path.join(out_dir, f"shard_{i}.npy")
        np.save(out_path, shard)
        
        if i % 5 == 0:
            print(f"[INDEX BUILDER] Info: Saved shard_{i}.npy ({shard.shape[0]} rows)")
        i += 1
        
    print(f"[INDEX BUILDER] Info: Successfully wrote {i} shards to {out_dir}")

# ===============================================================
# Print shard for inspection / debugging
# ===============================================================
def print_hist_info(shard: np.ndarray, name: str):
    """
    Print summary information about a single shard for inspection.
    """
    print(f"\n--- Info for {name} ---")
    print(f"Shape: {shard.shape}")
    print(f"Dtype: {shard.dtype}")
    # Chỉ in 2 dòng đầu để xem mẫu
    print(f"[INDEX BUILDER] Info: Preview (First 2 rows):\n{shard[:2]}")
    # Với float32, in min/max giúp kiểm tra xem vector có bị chuẩn hóa không
    print(f"[INDEX BUILDER] Info: Min={shard.min():.4f}, Max={shard.max():.4f}")
    print("----------------------")

def print_all_shards_info(shard_dir: str):
    """
    Iterate through all .npy shard files in a directory and print info for each.
    """
    shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".npy")])
    if not shard_files:
        print(f"No shard files found in {shard_dir}")
        return

    # Chỉ inspect tối đa 3 file đầu để đỡ rác màn hình nếu có quá nhiều shard
    print(f"[INDEX BUILDER] Info: Inspecting first 3 shards in {shard_dir}...")
    for fname in shard_files[:3]:
        path = os.path.join(shard_dir, fname)
        shard = np.load(path)
        print_hist_info(shard, fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help="Path to input .npy file (e.g., data/X.npy)")
    parser.add_argument('--out', required=True, help="Output directory for shards")
    parser.add_argument('--shard-size', type=int, default=100000, help="Number of rows per shard")
    parser.add_argument('--inspect', action='store_true', help="Print info of shards after splitting")
    args = parser.parse_args()

    split_and_save(args.data, args.out, shard_size=args.shard_size)

    if args.inspect:
        print_all_shards_info(args.out)