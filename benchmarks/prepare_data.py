# benchmarks/prepare_data.py
import os
import requests
import zipfile
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np # Chỉ dùng để tạo ID

# --- CHECK LIBRARIES ---
try:
    import pyarrow
except ImportError:
    print("[DATA-LITE] Error: Missing 'pyarrow'. Please run: pip install pyarrow")
    sys.exit(1)

try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
except ImportError:
    print("[DATA-LITE] Error: Missing 'gensim'. Please run: pip install gensim")
    sys.exit(1)

# --- CONFIGURATION ---
DATA_DIR = "data"

# URLs & Paths
GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_ZIP = os.path.join(DATA_DIR, "glove.840B.300d.zip")
GLOVE_TXT = os.path.join(DATA_DIR, "glove.840B.300d.txt")

W2V_LOCAL = os.path.join(DATA_DIR, "word2vec-google-news-300.kv")

# Output: Chỉ meta.parquet
META_PATH = os.path.join(DATA_DIR, "meta.parquet")

# ================== HELPER FUNCTIONS ==================

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def is_clean_word(w: str) -> bool:
    """Chỉ lấy từ thuần chữ cái, độ dài >= 2"""
    return w.isalpha() and len(w) >= 2

def download_and_extract_glove():
    ensure_data_dir()
    # 1. Download
    if not os.path.isfile(GLOVE_ZIP) and not os.path.isfile(GLOVE_TXT):
        print(f"[DATA-LITE] Downloading GloVe...")
        try:
            resp = requests.get(GLOVE_URL, stream=True)
            total_size = int(resp.headers.get("Content-Length", 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Download GloVe") as pbar:
                with open(GLOVE_ZIP, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        except Exception as e:
            print(f"[DATA-LITE] Error downloading GloVe: {e}")
            sys.exit(1)
            
    # 2. Extract
    if not os.path.isfile(GLOVE_TXT):
        print(f"[DATA-LITE] Extracting GloVe...")
        with zipfile.ZipFile(GLOVE_ZIP, "r") as z:
            z.extract("glove.840B.300d.txt", path=DATA_DIR)

# ================== MAIN LOGIC ==================

def make_dictionary_only():
    ensure_data_dir()
    
    words = []
    sources = []
    
    # ---------------------------------------------------------
    # 1. XỬ LÝ GLOVE (Chỉ lấy chữ)
    # ---------------------------------------------------------
    download_and_extract_glove()
    print(f"\n[DATA-LITE] [1/3] Scanning GloVe words...")
    
    try:
        # Tổng dòng GloVe 840B để hiện progress bar
        total_lines = 2196017 
        with open(GLOVE_TXT, "r", encoding="utf8", errors="ignore") as f:
            for line in tqdm(f, total=total_lines, desc="Reading GloVe"):
                # Chỉ cắt lấy từ đầu tiên, bỏ qua toàn bộ phần vector phía sau
                # split(' ', 1) giúp tách nhanh hơn split(' ')
                parts = line.split(' ', 1)
                word = parts[0]
                
                if is_clean_word(word):
                    words.append(word)
                    sources.append("glove")
                    
    except FileNotFoundError:
        print(f"[DATA-LITE] Error: File {GLOVE_TXT} not found")
        sys.exit(1)

    print(f"[DATA-LITE] GloVe words found: {len(words)}")

    # ---------------------------------------------------------
    # 2. XỬ LÝ WORD2VEC (Chỉ lấy chữ)
    # ---------------------------------------------------------
    print(f"\n[DATA-LITE] [2/3] Scanning Word2Vec words...")
    
    wv_exists = False
    if not os.path.isfile(W2V_LOCAL):
        print("[DATA-LITE] Downloading W2V model (metadata only needed)...")
        try:
            wv = api.load("word2vec-google-news-300")
            wv.save(W2V_LOCAL)
            wv_exists = True
        except:
            print("[DATA-LITE] Warning: Cannot download W2V. Skipping.")
    else:
        print("[DATA-LITE] Loading local W2V...")
        try:
            wv = KeyedVectors.load(W2V_LOCAL, mmap='r')
            wv_exists = True
        except:
            pass

    if wv_exists:
        existing_set = set(words)
        count_add = 0
        # wv.index_to_key chứa danh sách từ, không cần load vector
        for word in tqdm(wv.index_to_key, desc="Reading W2V"):
            if is_clean_word(word) and word not in existing_set:
                words.append(word)
                sources.append("w2v")
                count_add += 1
        print(f"[DATA-LITE] Added {count_add} new words from W2V.")

    # ---------------------------------------------------------
    # 3. LƯU PARQUET (ID, WORD, SOURCE)
    # ---------------------------------------------------------
    print(f"\n[DATA-LITE] [3/3] Saving dictionary to {META_PATH}...")
    
    df = pd.DataFrame({
        "id": np.arange(len(words), dtype=np.int32),
        "word": words,
        "source": sources
    })
    
    df.to_parquet(META_PATH, index=False)
    print(f"[DATA-LITE] Success! Saved {len(df)} words.")
    print(f"[DATA-LITE] Sample:\n{df.head()}")

if __name__ == "__main__":
    make_dictionary_only()