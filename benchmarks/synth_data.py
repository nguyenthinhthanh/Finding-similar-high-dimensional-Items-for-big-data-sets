# benchmarks/synth_data.py
import os
import numpy as np
import hashlib
import pickle
from typing import List, Set, Tuple
import argparse
import pandas as pd

# Prime for modular hashing
_PRIME = (1 << 61) - 1

def _stable_shingle_hash(sh: str) -> int:
    h = hashlib.sha1(sh.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % _PRIME

class MinHash:
    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = int(num_perm)
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, _PRIME - 1, size=self.num_perm, dtype=np.int64)
        self.b = rng.randint(0, _PRIME - 1, size=self.num_perm, dtype=np.int64)

    def signature(self, shingles: Set[str]) -> np.ndarray:
        if not shingles:
            return np.full(self.num_perm, _PRIME, dtype=np.uint64)

        sh_ints = np.array([_stable_shingle_hash(s) for s in shingles], dtype=np.int64)
        sig = np.empty(self.num_perm, dtype=np.uint64)
        for i in range(self.num_perm):
            vals = (int(self.a[i]) * sh_ints + int(self.b[i])) % _PRIME
            sig[i] = int(np.min(vals))
        return sig

    def batch_signature(self, shingles_list: List[Set[str]]) -> np.ndarray:
        sigs = [self.signature(s) for s in shingles_list]
        return np.vstack(sigs).astype(np.uint64)

# Shingling helpers
def shingle_document(doc: str, k: int = 5, by_word: bool = True) -> Set[str]:
    if doc is None:
        return set()
    if by_word:
        toks = doc.split()
        if len(toks) < k:
            return {" ".join(toks)}
        return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}
    else:
        s = doc
        if len(s) < k:
            return {s}
        return {s[i:i + k] for i in range(len(s) - k + 1)}

# Load real data produced by prepare_data (meta.parquet / meta_raw.parquet)
def load_prepare_data(meta_path: str = "data/meta.parquet",
                      use_raw: bool = False,
                      limit: int = None,
                      group_size: int = None,
                      shuffle: bool = False,
                      seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Load words from meta.parquet and return (docs, ids).
    Modes:
      - group_size is None or 1 => each word is a doc (doc_text = word)
      - group_size >1 => group consecutive words into a doc of group_size words (joined by space)
    If use_raw=True, use data/meta_raw.parquet instead.
    """
    path = "data/meta_raw.parquet" if use_raw else meta_path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Meta file not found: {path}")

    df = pd.read_parquet(path)
    words = df["word"].astype(str).tolist()

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(words)

    if limit is not None:
        words = words[:limit]

    docs = []
    ids = []
    if group_size is None or group_size <= 1:
        # Each word becomes a document
        docs = words
        # Use id from meta if present, otherwise create synthetic ids
        if "id" in df.columns:
            ids = df["id"].astype(str).tolist()[:len(docs)]
        else:
            ids = [f"w_{i:06d}" for i in range(len(docs))]
    else:
        # Group words into documents of group_size
        for i in range(0, len(words), group_size):
            chunk = words[i:i + group_size]
            docs.append(" ".join(chunk))
            ids.append(f"grp_{i//group_size:06d}")

    return docs, ids

# Synthetic generator (kept for fallback)
def make_synthetic_docs(n_docs: int = 10000,
                        vocab_size: int = 1000,
                        avg_words: int = 50,
                        sigma_words: float = 10.0,
                        out_dir: str = "data",
                        seed: int = 42) -> Tuple[List[str], List[str]]:
    rng = np.random.RandomState(seed)
    os.makedirs(out_dir, exist_ok=True)
    vocab = [f"w{idx}" for idx in range(vocab_size)]
    docs = []
    ids = []
    for i in range(n_docs):
        n_words = max(1, int(rng.normal(loc=avg_words, scale=sigma_words)))
        words = rng.choice(vocab, size=n_words, replace=True)
        docs.append(" ".join(words))
        ids.append(f"doc_{i:06d}")
    with open(os.path.join(out_dir, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)
    with open(os.path.join(out_dir, "ids.pkl"), "wb") as f:
        pickle.dump(ids, f)
    print(f"Saved {n_docs} synthetic docs to {out_dir}/docs.pkl and ids to ids.pkl")
    return docs, ids

def build_and_save_minhash_signatures(docs: List[str],
                                     ids: List[str],
                                     num_perm: int = 128,
                                     k_shingle: int = 3,
                                     by_word: bool = True,
                                     out_dir: str = "data",
                                     save_shingles: bool = True,
                                     seed: int = 42) -> np.ndarray:
    os.makedirs(out_dir, exist_ok=True)
    shingles_list = [shingle_document(d, k=k_shingle, by_word=by_word) for d in docs]
    mh = MinHash(num_perm=num_perm, seed=seed)
    sigs = mh.batch_signature(shingles_list)
    np.save(os.path.join(out_dir, "sigs.npy"), sigs)
    with open(os.path.join(out_dir, "ids.pkl"), "wb") as f:
        pickle.dump(ids, f)
    with open(os.path.join(out_dir, "minhash_meta.pkl"), "wb") as f:
        pickle.dump({"num_perm": num_perm, "k_shingle": k_shingle, "by_word": by_word, "seed": seed}, f)
    if save_shingles:
        with open(os.path.join(out_dir, "shingles.pkl"), "wb") as f:
            pickle.dump(shingles_list, f)
    print(f"Saved signatures to {out_dir}/sigs.npy (shape={sigs.shape}), metadata/minhash_meta.pkl")
    return sigs

def inspect_signatures(sigs: np.ndarray, docs: List[str], ids: List[str], n_sample: int = 5):
    print("Signatures stats:")
    print(f" - shape: {sigs.shape}")
    print(f" - dtype: {sigs.dtype}")
    print(f" - sample rows (first {n_sample}):")
    print(sigs[:n_sample])
    print("\nSample documents:")
    for i in range(min(n_sample, len(docs))):
        print(f" - id={ids[i]} len(doc)={len(docs[i])} -> {docs[i][:120]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate (or load) docs and build MinHash signatures.")
    parser.add_argument("--use-prepare", action="store_true", help="Load words from prepare_data (meta.parquet).")
    parser.add_argument("--use-raw", action="store_true", help="If using prepare data, load meta_raw.parquet instead of meta.parquet.")
    parser.add_argument("--meta-path", default="data/meta.parquet", help="Path to meta.parquet (prepare_data output).")
    parser.add_argument("--group-size", type=int, default=1, help="If >1, group N words into one document.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of words/docs to use (for testing).")
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--k-shingle", type=int, default=3)
    parser.add_argument("--by-word", action="store_true", help="Shingle by word (default off if not provided).")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make-synth", action="store_true", help="Force create synthetic docs instead of loading prepare_data.")
    args = parser.parse_args()

    if args.use_prepare and not args.make_synth:
        print("[INFO] Loading real words from prepare_data...")
        docs, ids = load_prepare_data(meta_path=args.meta_path, use_raw=args.use_raw,
                                      limit=args.limit, group_size=args.group_size,
                                      shuffle=False, seed=args.seed)
    else:
        # fallback to synthetic generator
        print("[INFO] Creating synthetic docs (fallback)...")
        docs, ids = make_synthetic_docs(n_docs=20000, vocab_size=20, avg_words=40,
                                        sigma_words=10.0, out_dir=args.out_dir, seed=args.seed)

    print("[INFO] Building MinHash signatures...")
    sigs = build_and_save_minhash_signatures(docs, ids,
                                            num_perm=args.num_perm,
                                            k_shingle=args.k_shingle,
                                            by_word=args.by_word,
                                            out_dir=args.out_dir,
                                            save_shingles=True,
                                            seed=args.seed)
    inspect_signatures(sigs, docs, ids, n_sample=5)
