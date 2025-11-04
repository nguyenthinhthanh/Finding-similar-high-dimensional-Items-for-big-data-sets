# benchmarks/synth_data.py
import os, requests, zipfile
import numpy as np
import gensim.downloader as api
from tqdm import tqdm
# ===============================================================
# This script builds a synthetic benchmark dataset by combining
# two large-scale pre-trained word embedding sources:
#   1. GloVe 840B.300d (from Stanford NLP)
#   2. Word2Vec Google News 300d (via gensim)
#
# The generated dataset (saved as `data/raw.npy`) can be used
# for benchmarking similarity search, clustering, or vector indexing.
#
# Main Steps:
#   (1) Download GloVe zip file if not already present
#   (2) Extract GloVe vectors
#   (3) Load pre-trained Word2Vec embeddings via Gensim
#   (4) Merge both sources into a single NumPy array
#   (5) Save either a full or reduced version for testing
# ===============================================================

# Data generation modes
DATA_MINIMAL_MODE = 0           # Generate smaller synthetic dataset (for testing)
DATA_FULL_MODE = 1              # Use full dataset (GloVe + Word2Vec)
DATA_MODE = DATA_MINIMAL_MODE

def make_data():
    """Create or load combined word embedding dataset (GloVe + Word2Vec)."""

    # === Define paths and URLs ===
    GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
    GLOVE_ZIP = "data/glove.840B.300d.zip"
    GLOVE_TXT = "data/glove.840B.300d.txt"

    # === 1. Download GloVe zip if missing ===
    if not os.path.isfile(GLOVE_ZIP):
        print(f"Downloading {GLOVE_ZIP} from {GLOVE_URL} ...")
        resp = requests.get(GLOVE_URL, stream=True)
        total_size = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading GloVe", ascii=".=") as pbar:
            with open(GLOVE_ZIP, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"Download completed: {GLOVE_ZIP}")

    # === 2. Extract GloVe text file if missing ===
    if not os.path.isfile(GLOVE_TXT):
        # print(f"Extracting {GLOVE_ZIP} ...")
        with zipfile.ZipFile(GLOVE_ZIP, "r") as z:
            z.extract("glove.840B.300d.txt", path="data/")
        # print("Extraction glove.840B.300d.txt completed.")

    # === 3. Load Word2Vec (Google News 300d) using Gensim ===
    print("Loading Word2Vec Google News 300d via Gensim...")
    model = api.load("word2vec-google-news-300")
    print("Word2Vec model successfully loaded.")

    # === 4. Count vocabulary sizes for both models ===
    # print("Counting GloVe vocabulary...")
    glove_count = 0
    with open(GLOVE_TXT, "r", encoding="utf-8") as f:
        for _ in f:
            glove_count += 1
    # print(f"GloVe word count: {glove_count}")

    w2v_count = len(model.index_to_key)
    # print(f"Word2Vec word count: {w2v_count}")

    total_count = glove_count + w2v_count
    dim = 300

    # === 5. Allocate memory for embeddings and metadata ===
    X = np.empty((total_count, dim), dtype=np.float32)
    words = [None] * total_count
    sources = [None] * total_count

    # === 6. Read GloVe vectors into memory ===
    # print("Reading and processing GloVe vectors...")
    with open(GLOVE_TXT, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=glove_count, desc="Processing GloVe", ascii=".=")):
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            words[i] = parts[0]
            sources[i] = "glove"
            X[i, :] = np.fromiter(parts[1:], dtype=np.float32)

    # === 7. Append Word2Vec vectors ===
    # print("Merging Word2Vec vectors into array...")
    start_idx = glove_count
    batch_size = 100000
    for j in tqdm(range(0, w2v_count, batch_size), desc="Processing Word2Vec", ascii=".="):
        end_j = min(j + batch_size, w2v_count)
        batch_words = model.index_to_key[j:end_j]
        batch_vecs = model.vectors[j:end_j]
        X[start_idx + j : start_idx + end_j, :] = batch_vecs
        for k, w in enumerate(batch_words):
            words[start_idx + j + k] = w
            sources[start_idx + j + k] = "word2vec"

    if DATA_MODE == DATA_FULL_MODE:
        print(f"Data mode: {DATA_MODE} → using full dataset.")
        np.save("data/raw.npy", X)
        print(f"Saved full dataset raw.npy (shape={X.shape}).")
        return X
    else:
        print(f"Data mode: {DATA_MODE} → using reduced dataset.")
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], size=100_000, replace=False)
        Y = X[indices]
        np.save('data/raw.npy', Y)
        print(f"Saved reduced dataset raw.npy (shape={Y.shape}).")
        return Y

def inspect_data(X):
    """Display basic statistics and sample rows of the dataset."""
    print("\nData inspection:")
    print(f"- Shape: {X.shape}")
    print(f"- Mean: {np.mean(X):.4f}")
    print(f"- Std deviation: {np.std(X):.4f}")
    print(f"- Min value: {np.min(X):.4f}")
    print(f"- Max value: {np.max(X):.4f}")
    
    print("\nSample rows:")
    print(X[:5])

# ===============================================================
# Entry point
# ===============================================================
if __name__ == '__main__':
    data = make_data()
    # Uncomment below to inspect generated dataset
    inspect_data(data)