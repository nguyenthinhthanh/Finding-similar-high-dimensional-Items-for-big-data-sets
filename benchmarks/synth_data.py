# benchmarks/synth_data.py
import os
import sys
import numpy as np
import pandas as pd
import argparse
import pickle
import gc
from tqdm import tqdm

# --- CẤU HÌNH ---
# Input dimension của GloVe và Word2Vec đều là 300
INPUT_DIM = 300   
# Output dimension (Signature size) - Giữ 128 để khớp hệ thống cũ
NUM_BITS = 128    

# Đường dẫn mặc định (giống prepare_data.py)
DATA_DIR = "data"
GLOVE_TXT = os.path.join(DATA_DIR, "glove.840B.300d.txt")
W2V_LOCAL = os.path.join(DATA_DIR, "word2vec-google-news-300.kv")

# Thư viện bổ trợ
try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
except ImportError:
    pass

def is_clean_word(w: str) -> bool:
    """Logic lọc từ gốc của bạn: Chỉ lấy chữ cái, độ dài >= 2"""
    return w.isalpha() and len(w) >= 2

def generate_simhash_batch(vectors, planes):
    """
    SimHash: Chiếu vector lên các mặt phẳng ngẫu nhiên.
    Input: (N, 300)
    Output: (N, 128) dạng 0/1 (uint64)
    """
    # Dot product: (N, 300) x (300, 128) -> (N, 128)
    projections = np.dot(vectors, planes)
    # Lượng tử hóa: > 0 là 1, <= 0 là 0
    return (projections > 0).astype(np.uint64)

def main(args):
    # 1. Kiểm tra dữ liệu đầu vào
    if args.use_prepare:
        if not os.path.exists(GLOVE_TXT):
             print(f"[ERROR] Không thấy {GLOVE_TXT}. Hãy chạy 'python benchmarks/prepare_data.py' trước.")
             sys.exit(1)

    # 2. Khởi tạo mặt phẳng chiếu (Projection Matrix) cho SimHash
    # Đây là "chìa khóa" để biến Vector 300d -> Signature 128d
    np.random.seed(42)
    planes = np.random.randn(INPUT_DIM, NUM_BITS).astype(np.float32)

    words = []
    vectors_list = []
    seen_words = set()
    
    # Mặc định lấy 1.5 triệu từ nếu không set limit
    limit_count = args.limit if (args.limit and args.limit > 0) else 1600000
    
    # Dictionary nhỏ để lưu Top 50k từ (dùng cho Service tra cứu nhanh)
    mini_vocab = {}

    # ---------------------------------------------------------
    # GIAI ĐOẠN 1: ĐỌC GLOVE (Nguồn chính)
    # ---------------------------------------------------------
    print(f"\n[1/2] Đang xử lý GloVe...")
    with open(GLOVE_TXT, "r", encoding="utf8", errors="ignore") as f:
        for line in tqdm(f, total=2196017, desc="Scanning GloVe"):
            if len(words) >= limit_count: break
            
            parts = line.rstrip().split(" ")
            word = parts[0]
            
            # Lọc rác
            if not is_clean_word(word): continue
            if word in seen_words: continue
            
            try:
                # Lấy vector 300 chiều
                vec = np.asarray(parts[1:], dtype=np.float32)
                if vec.shape[0] == INPUT_DIM:
                    words.append(word)
                    vectors_list.append(vec)
                    seen_words.add(word)
                    
                    # Lưu vào mini_vocab nếu còn chỗ (để Service dùng)
                    if len(mini_vocab) < 50000:
                        mini_vocab[word] = vec
            except: 
                continue

    print(f"-> Đã lấy {len(words)} từ từ GloVe.")

    # ---------------------------------------------------------
    # GIAI ĐOẠN 2: BỔ SUNG WORD2VEC (Nguồn phụ)
    # ---------------------------------------------------------
    # Chỉ chạy nếu chưa đủ limit và có cài gensim
    if 'gensim' in sys.modules and len(words) < limit_count:
        print(f"\n[2/2] Đang kiểm tra Word2Vec (để bổ sung)...")
        
        # Logic tải/load model W2V y hệt file cũ của bạn
        has_w2v = False
        wv = None
        
        if os.path.exists(W2V_LOCAL):
            print("-> Loading local W2V...")
            try:
                wv = KeyedVectors.load(W2V_LOCAL, mmap='r')
                has_w2v = True
            except: pass
        else:
            print("-> Downloading W2V (lần đầu)...")
            try:
                wv = api.load("word2vec-google-news-300")
                wv.save(W2V_LOCAL)
                has_w2v = True
            except: pass

        if has_w2v and wv:
            print("-> Merging W2V words...")
            for word in tqdm(wv.index_to_key, desc="Merging W2V"):
                if len(words) >= limit_count: break
                
                # Logic lọc và check trùng
                if is_clean_word(word) and word not in seen_words:
                    vec = wv[word]
                    if vec.shape[0] == INPUT_DIM:
                        words.append(word)
                        vectors_list.append(vec.astype(np.float32))
                        seen_words.add(word)
                        
                        # Bổ sung vào mini_vocab nếu chưa đầy
                        if len(mini_vocab) < 50000:
                            mini_vocab[word] = vec.astype(np.float32)
            
            # Giải phóng RAM
            del wv
            gc.collect()

    # ---------------------------------------------------------
    # GIAI ĐOẠN 3: TÍNH SIMHASH & LƯU
    # ---------------------------------------------------------
    if not vectors_list:
        print("[ERROR] Không tìm thấy từ nào hợp lệ!")
        sys.exit(1)

    N = len(words)
    print(f"\n[3/3] Đang tính SimHash cho {N} từ...")
    
    # Gom list thành Matrix lớn (N, 300)
    X_float = np.vstack(vectors_list)
    
    # Tính Signature (N, 128)
    X_sigs = generate_simhash_batch(X_float, planes)
    
    # Giải phóng vector float để tiết kiệm RAM (chỉ giữ lại sigs uint64)
    del X_float
    del vectors_list
    gc.collect()

    # Lưu file vector (sigs.npy)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    sigs_path = os.path.join(out_dir, "sigs.npy")
    print(f"-> Lưu {sigs_path}...")
    np.save(sigs_path, X_sigs)
    
    # Lưu file Meta (meta.parquet) - chứa ID và Word
    meta_path = args.meta_path if args.meta_path else os.path.join(out_dir, "meta.parquet")
    df = pd.DataFrame({"id": np.arange(N).astype(str), "word": words})
    df.to_parquet(meta_path, index=False)
    print(f"-> Lưu {meta_path}")

    # Lưu Semantic Config (Quan trọng cho Service)
    # Service cần 'planes' để hash câu query, và 'mini_vocab' để tra từ
    config_path = os.path.join(out_dir, "semantic_config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump({
            "projections": planes, 
            "mini_vocab": mini_vocab
        }, f)
    print(f"-> Lưu {config_path} (Dùng cho Query Service)")

    # Lưu file giả minhash_meta để tương thích ngược (nếu cần)
    with open(os.path.join(out_dir, "minhash_meta.pkl"), "wb") as f:
        pickle.dump({"num_perm": NUM_BITS, "semantic": True}, f)

    print("\n=== HOÀN TẤT QUÁ TRÌNH TẠO DỮ LIỆU SEMANTIC ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-prepare", action="store_true", default=True, help="Dùng dữ liệu từ prepare_data")
    parser.add_argument("--limit", type=int, default=1500000, help="Giới hạn số từ (Mặc định 1.5 triệu)")
    parser.add_argument("--meta-path", default="data/meta.parquet")
    parser.add_argument("--out-dir", default="data")
    # Các tham số cũ giữ lại để không lỗi lệnh gọi
    parser.add_argument("--k-shingle", type=int, default=3)
    parser.add_argument("--by-word", action="store_true")

    args = parser.parse_args()
    main(args)