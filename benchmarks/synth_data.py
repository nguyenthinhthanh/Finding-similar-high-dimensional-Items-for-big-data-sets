# benchmarks/synth_data.py
import os
import requests
import zipfile
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- KIỂM TRA THƯ VIỆN ---
try:
    import pyarrow
except ImportError:
    print("LỖI: Thiếu thư viện 'pyarrow'. Hãy chạy: pip install pyarrow")
    sys.exit(1)

try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
except ImportError:
    print("LỖI: Thiếu thư viện 'gensim'. Hãy chạy: pip install gensim")
    sys.exit(1)

# ================== CẤU HÌNH ==================

DATA_DIR = "data"

# URLs & Paths
GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_ZIP = os.path.join(DATA_DIR, "glove.840B.300d.zip")
GLOVE_TXT = os.path.join(DATA_DIR, "glove.840B.300d.txt")

W2V_LOCAL = os.path.join(DATA_DIR, "word2vec-google-news-300.kv")

# Output Paths
X_PATH = os.path.join(DATA_DIR, "X.npy")            # Data chính (Full)
META_PATH = os.path.join(DATA_DIR, "meta.parquet")  # Từ điển (Full)

RAW_PATH = os.path.join(DATA_DIR, "raw.npy")        # Data test (Subset)
RAW_META_PATH = os.path.join(DATA_DIR, "meta_raw.parquet")

# ================== HÀM HỖ TRỢ ==================

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def is_clean_word(w: str) -> bool:
    """
    Bộ lọc chất lượng:
    - Chỉ lấy từ toàn chữ cái (A-Z, a-z).
    - Độ dài >= 2 (bỏ các từ như 'a', 'I', 'k'...).
    """
    return w.isalpha() and len(w) >= 2

def download_glove():
    ensure_data_dir()
    if not os.path.isfile(GLOVE_ZIP) and not os.path.isfile(GLOVE_TXT):
        print(f"-> Đang tải GloVe từ {GLOVE_URL}...")
        try:
            resp = requests.get(GLOVE_URL, stream=True)
            total_size = int(resp.headers.get("Content-Length", 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading GloVe", ascii=".=") as pbar:
                with open(GLOVE_ZIP, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            print("-> Tải GloVe hoàn tất.")
        except Exception as e:
            print(f"Lỗi khi tải GloVe: {e}")
            sys.exit(1)

def extract_glove():
    ensure_data_dir()
    if os.path.isfile(GLOVE_TXT):
        print(f"-> Đã có {GLOVE_TXT}, bỏ qua giải nén.")
        return
    if not os.path.isfile(GLOVE_ZIP):
        print(f"LỖI: Không tìm thấy {GLOVE_ZIP}.")
        sys.exit(1)
        
    print(f"-> Đang giải nén {GLOVE_ZIP}...")
    with zipfile.ZipFile(GLOVE_ZIP, "r") as z:
        z.extract("glove.840B.300d.txt", path=DATA_DIR)
    print("-> Giải nén xong.")

# ================== LOGIC CHÍNH (FULL DATA) ==================

def make_data() -> np.ndarray:
    """
    Tạo bộ dữ liệu FULL (Không giới hạn số lượng).
    """
    ensure_data_dir()

    # 1. Xử lý GloVe
    download_glove()
    extract_glove()

    words = []
    sources = []
    vectors = []

    print(f"\n[1/4] Đang đọc TOÀN BỘ GloVe (có thể mất thời gian)...")
    try:
        # Số dòng chuẩn của glove.840B.300d để hiển thị thanh tiến trình
        total_lines = 2196017 
        
        with open(GLOVE_TXT, "r", encoding="utf8", errors="ignore") as f:
            for line in tqdm(f, total=total_lines, desc="Processing GloVe"):
                parts = line.rstrip().split(" ")
                if len(parts) < 301: continue 
                
                word = parts[0]
                if not is_clean_word(word): continue
                
                try:
                    vec = np.asarray(parts[1:], dtype=np.float32)
                except ValueError: continue
                
                if vec.shape[0] != 300: continue
                
                words.append(word)
                sources.append("glove")
                vectors.append(vec)
    except FileNotFoundError:
        print(f"Lỗi: Không đọc được file {GLOVE_TXT}")
        sys.exit(1)

    print(f"-> Đã lấy {len(words)} từ GloVe.")

    # 2. Xử lý Word2Vec (Google News)
    print(f"\n[2/4] Đang xử lý TOÀN BỘ Word2Vec...")
    wv = None
    if not os.path.isfile(W2V_LOCAL):
        print("-> Đang tải model word2vec-google-news-300 (khoảng 1.6GB)...")
        try:
            wv = api.load("word2vec-google-news-300")
            wv.save(W2V_LOCAL)
        except Exception as e:
            print(f"CẢNH BÁO: Không tải được Word2Vec ({e}). Chỉ dùng GloVe.")
    else:
        print(f"-> Load Word2Vec từ cache local: {W2V_LOCAL}")
        try:
            wv = KeyedVectors.load(W2V_LOCAL, mmap='r')
        except Exception as e:
            print(f"Lỗi load cache Word2Vec: {e}. Bỏ qua W2V.")

    if wv:
        existing_words = set(words)
        added_count = 0
        for word in tqdm(wv.index_to_key, desc="Processing Word2Vec"):
            if not is_clean_word(word): continue
            if word in existing_words: continue 
            
            vec = wv[word]
            if vec.shape[0] != 300: continue
            
            words.append(word)
            sources.append("w2v")
            vectors.append(vec.astype(np.float32))
            added_count += 1
        print(f"-> Đã thêm {added_count} từ từ Word2Vec.")

    # 3. Ghép và Lưu
    print("\n[3/4] Đang ghép dữ liệu (Cẩn thận tràn RAM)...")
    if len(vectors) == 0:
        print("LỖI: Không tìm thấy vector nào hợp lệ!")
        sys.exit(1)

    # Bước tốn RAM nhất: vstack
    X = np.vstack(vectors).astype(np.float32)
    print(f"-> Final Shape X: {X.shape}, Size: {X.nbytes / 1024**3:.2f} GB")
    
    print("-> Đang lưu file X.npy (sẽ mất vài giây)...")
    np.save(X_PATH, X)
    
    print("-> Đang lưu file meta.parquet...")
    df_meta = pd.DataFrame({
        "word": words,
        "source": sources,
        "id": np.arange(len(words), dtype=np.int32)
    })
    df_meta.to_parquet(META_PATH, index=False)
    print(f"-> Đã xong dữ liệu chính.")

    # 4. Tạo tập test nhỏ
    print("\n[4/4] Đang tạo tập test nhỏ (raw.npy) giữ nguyên 300 chiều...")
    # Lấy tối đa 10k mẫu để test
    sample_size = min(10000, X.shape[0])
    
    np.random.seed(42)
    idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
    
    Y = X[idx, :] 
    np.save(RAW_PATH, Y)
    
    df_meta_raw = df_meta.iloc[idx].copy()
    df_meta_raw["orig_id"] = df_meta_raw["id"]
    df_meta_raw["id"] = np.arange(len(df_meta_raw), dtype=np.int32)
    df_meta_raw.to_parquet(RAW_META_PATH, index=False)
    
    print(f"-> Đã lưu tập test: {RAW_PATH}")
    print("\n=== HOÀN TẤT: DỮ LIỆU ĐÃ SẴN SÀNG ===")
    
    return X

# ================== MAIN ==================

if __name__ == "__main__":
    make_data()