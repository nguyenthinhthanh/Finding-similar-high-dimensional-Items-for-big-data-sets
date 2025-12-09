# benchmarks/prepare_data.py
import os
import requests
import zipfile
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CHECK LIBRARIES ---
try:
    import pyarrow
except ImportError:
    print("[PP DATA] Error: Missing 'pyarrow' library. Please run: pip install pyarrow")
    sys.exit(1)

try:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
except ImportError:
    print("[PP DATA] Error: Missing 'gensim' library. Please run: pip install gensim")
    sys.exit(1)

# --- CONFIGURATION ---
DATA_DIR = "data"

# URLs & Paths
GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_ZIP = os.path.join(DATA_DIR, "glove.840B.300d.zip")
GLOVE_TXT = os.path.join(DATA_DIR, "glove.840B.300d.txt")

W2V_LOCAL = os.path.join(DATA_DIR, "word2vec-google-news-300.kv")

# Output Paths
X_PATH = os.path.join(DATA_DIR, "X.npy")            # Main data (Full) - 1.6GB
META_PATH = os.path.join(DATA_DIR, "meta.parquet")  # Dictionary (Full)

RAW_PATH = os.path.join(DATA_DIR, "raw.npy")        # Test data (Subset)
RAW_META_PATH = os.path.join(DATA_DIR, "meta_raw.parquet")

# ================== HELPER FUNCTIONS ==================

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def is_clean_word(w: str) -> bool:
    """
    Quality filter:
    - Only take words consisting entirely of letters (A-Z, a-z).
    - Length >= 2 (exclude words like 'a', 'I', 'k'...).
    """
    return w.isalpha() and len(w) >= 2

def download_glove():
    ensure_data_dir()
    if not os.path.isfile(GLOVE_ZIP) and not os.path.isfile(GLOVE_TXT):
        print(f"[PP DATA] Info: Downloading GloVe from {GLOVE_URL}...")
        try:
            resp = requests.get(GLOVE_URL, stream=True)
            total_size = int(resp.headers.get("Content-Length", 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading GloVe", ascii=".=") as pbar:
                with open(GLOVE_ZIP, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            print("[PP DATA] Info: Downloading GloVe completed.")
        except Exception as e:
            print(f"[PP DATA] Error: downloading GloVe: {e}")
            sys.exit(1)

def extract_glove():
    ensure_data_dir()
    if os.path.isfile(GLOVE_TXT):
        print(f"[PP DATA] Info: {GLOVE_TXT} already exists, skipping extraction.")
        return
    if not os.path.isfile(GLOVE_ZIP):
        print(f"[PP DATA] Error: {GLOVE_ZIP} not found.")
        sys.exit(1)
        
    print(f"[PP DATA] Info: Extracting {GLOVE_ZIP}...")
    with zipfile.ZipFile(GLOVE_ZIP, "r") as z:
        z.extract("glove.840B.300d.txt", path=DATA_DIR)
    print("[PP DATA] Info: Extraction completed.")

# ================== MAIN LOGIC (FULL DATA) ==================

def make_data() -> np.ndarray:
    """
    Create FULL dataset (No size limit).
    """
    ensure_data_dir()

    # 1. Process GloVe
    download_glove()
    extract_glove()

    words = []
    sources = []
    vectors = []

    print(f"\n[PP DATA] Info:[1/4] Reading FULL GloVe (may take time)...")
    try:
        # Standard line count of glove.840B.300d for progress bar
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
        print(f"[PP DATA] Error: Cannot read file {GLOVE_TXT}")
        sys.exit(1)

    print(f"[PP DATA] Info: Retrieved {len(words)} words from GloVe.")

    # 2. Process Word2Vec (Google News)
    print(f"\n[PP DATA] Info:[2/4] Processing FULL Word2Vec...")
    wv = None
    if not os.path.isfile(W2V_LOCAL):
        print(f"[PP DATA] Info: Downloading word2vec-google-news-300 model (approx. 1.6GB)...")
        try:
            wv = api.load("word2vec-google-news-300")
            wv.save(W2V_LOCAL)
        except Exception as e:
            print(f"[PP DATA] Warning: Could not download Word2Vec ({e}). Using GloVe only.")
    else:
        print(f"[PP DATA] Info: Loading Word2Vec from local cache: {W2V_LOCAL}")
        try:
            wv = KeyedVectors.load(W2V_LOCAL, mmap='r')
        except Exception as e:
            print(f"[PP DATA] Error: loading Word2Vec cache: {e}. Skipping W2V.")

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
        print(f"[PP DATA] Info: Added {added_count} words from Word2Vec.")

    # 3. Merge and Save
    print("\n[PP DATA] Info:[3/4] Merging data (Be careful of RAM overflow)...")
    if len(vectors) == 0:
        print("[PP DATA] Error: No valid vectors found!")
        sys.exit(1)

    # Most RAM-intensive step: vstack
    X = np.vstack(vectors).astype(np.float32)
    print(f"[PP DATA] Info: Final Shape X: {X.shape}, Size: {X.nbytes / 1024**3:.2f} GB")
    
    print("[PP DATA] Info: Saving file X.npy (this may take a few seconds)...")
    np.save(X_PATH, X)
    
    print("[PP DATA] Info: Saving file meta.parquet...")
    df_meta = pd.DataFrame({
        "word": words,
        "source": sources,
        "id": np.arange(len(words), dtype=np.int32)
    })
    df_meta.to_parquet(META_PATH, index=False)
    print("[PP DATA] Info: Main data saved.")

    # 4. Creating small test set
    print("\n[PP DATA] Info:[4/4] Creating small test set (raw.npy) with original 300 dimensions...")
    # Take up to 10k samples for testing
    sample_size = min(10000, X.shape[0])
    
    np.random.seed(42)
    idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
    
    Y = X[idx, :] 
    np.save(RAW_PATH, Y)
    
    df_meta_raw = df_meta.iloc[idx].copy()
    df_meta_raw["orig_id"] = df_meta_raw["id"]
    df_meta_raw["id"] = np.arange(len(df_meta_raw), dtype=np.int32)
    df_meta_raw.to_parquet(RAW_META_PATH, index=False)
    
    print(f"[PP DATA] Info: Saved test set: {RAW_PATH}")
    print("\n[PP DATA] Info: === COMPLETED: DATA IS READY ===")
    
    return X

# ================== MAIN ==================

if __name__ == "__main__":
    make_data()