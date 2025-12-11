# app_client.py
import pandas as pd
import numpy as np
import requests
import os
from tqdm import tqdm

# CONFIG
DATA_DIR = "data"
META_PATH = os.path.join(DATA_DIR, "meta.parquet")  # chứa cột 'id' (string) và 'word'
VEC_PATH = os.path.join(DATA_DIR, "sigs.npy")       # chứa signature matrix (N,128) 0/1
QUERY_URL = "http://localhost:8000/query"
SHARD_SIZE = 5000

class Vocabulary:
    def __init__(self, meta_path: str):
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        df = pd.read_parquet(meta_path)
        # meta.parquet expected columns: 'id' and 'word'
        # keep maps both directions
        self.word2id = {}
        self.id2word = {}
        # try to coerce id -> integer global id if stored as strings of numbers
        for i, row in df.iterrows():
            idx = row['id']
            word = row['word']
            # try numeric cast if possible (keep original if not)
            try:
                idx_int = int(idx)
            except Exception:
                # if id is like "w0000001" keep numeric position instead
                idx_int = i
            self.word2id[word] = idx_int
            self.id2word[idx_int] = word

    def get_id(self, word: str):
        return self.word2id.get(word)

    def get_word(self, global_id: int):
        return self.id2word.get(global_id, "<unk>")

class VectorStore:
    def __init__(self, vec_path):
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"Vector file not found: {vec_path}")
        # print(f"[APP CLIENT] Loading vectors from {vec_path} (mmap_mode='r') ...")
        self.data = np.load(vec_path, mmap_mode='r')
        # print(f"[APP CLIENT] Vector shape: {self.data.shape}, dtype={self.data.dtype}")

    def get_vector(self, idx):
        if idx is None:
            return None
        if 0 <= idx < self.data.shape[0]:
            return self.data[idx]
        return None

def send_query_vector(vec, k=10):
    """
    vec: numpy array (signature) or list
    k: top-k
    """
    payload = {
        "vector": vec.tolist() if isinstance(vec, np.ndarray) else list(vec),
        "k": int(k)
    }
    resp = requests.post(QUERY_URL, json=payload, timeout=30)
    return resp

def parse_result_id(res_id):
    """
    res_id could be:
      - list/tuple [shard_idx, row_idx]
      - int global_id
      - string (we try to parse int)
    Returns global_id (int) or None
    """
    if isinstance(res_id, (list, tuple)) and len(res_id) == 2:
        shard_idx, row_idx = res_id
        try:
            shard_idx = int(shard_idx); row_idx = int(row_idx)
            return shard_idx * SHARD_SIZE + row_idx
        except Exception:
            return None
    # if it's an int-like
    try:
        return int(res_id)
    except Exception:
        return None

def pretty_print_results(results_json, vocab: Vocabulary):
    """
    results_json: list of { "id": ..., "score": ..., "vector_preview": [...] }
    """
    print("-" * 80)
    print(f"{'RANK':<6}{'WORD':<30}{'SCORE':<12}{'GLOBAL_ID':<12}{'SHARD/ROW':<15}")
    print("-" * 80)
    for i, r in enumerate(results_json):
        rid = r.get("id")
        score = r.get("score", 0.0)
        # try derive global id
        global_id = parse_result_id(rid)
        # derive shard/row string for display
        shard_row = ""
        if isinstance(rid, (list, tuple)) and len(rid) == 2:
            shard_row = f"{rid[0]}/{rid[1]}"
        else:
            shard_row = "-"
        word = vocab.get_word(global_id) if global_id is not None else "<unknown>"
        print(f"#{i+1:<5}{word:<30}{score:<12.6f}{str(global_id):<12}{shard_row:<15}")
    print("-" * 80)

def main():
    # check files
    if not os.path.exists(META_PATH) or not os.path.exists(VEC_PATH):
        print("[APP CLIENT] Error: Required data files not found.")
        print("Please run the data preparation step (prepare_data/synth_data) first.")
        return

    vocab = Vocabulary(META_PATH)
    store = VectorStore(VEC_PATH)

    print("\n--- SEARCH CLIENT READY (type 'exit' to quit) ---")
    while True:
        word = input("\nEnter word to search (or 'exit'): ").strip()
        if word.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if word == "":
            continue

        k_input = input("Enter k (top-k, default 5): ").strip()
        try:
            k = int(k_input) if k_input else 5
            if k <= 0:
                print("[APP CLIENT] k must be positive integer. Using k=5.")
                k = 5
        except Exception:
            print("[APP CLIENT] Invalid k, using default k=5.")
            k = 5

        # find word id
        word_id = vocab.get_id(word)
        if word_id is None:
            print(f"[APP CLIENT] Word '{word}' not found in meta vocabulary.")
            # Also show suggestions (prefix match)
            suggestions = [w for w in vocab.word2id.keys() if w.startswith(word[:3])]
            if suggestions:
                print("Did you mean:", ", ".join(suggestions[:10]))
            continue

        # get vector
        vec = store.get_vector(word_id)
        if vec is None:
            print("[APP CLIENT] Vector for that id not found.")
            continue

        try:
            resp = send_query_vector(vec, k=k)
        except Exception as e:
            print(f"[APP CLIENT] Error connecting to server: {e}")
            continue

        if resp.status_code != 200:
            print(f"[APP CLIENT] Server returned error {resp.status_code}: {resp.text}")
            continue

        resp_json = resp.json()
        candidates = resp_json.get("candidates", [])
        if not candidates:
            print("[APP CLIENT] No candidates returned.")
            continue

        print(f"\nTop-{k} similar to '{word}':")
        pretty_print_results(candidates, vocab)

if __name__ == "__main__":
    main()
