# benchmarks/synth_data.py
import os
import numpy as np
import hashlib
import pickle
from typing import List, Set, Tuple
"""
Generate synthetic *documents* and MinHash signatures (for Jaccard similarity).
Saves:
 - data/sigs.npy      : numpy array shape (n_docs, num_perm) dtype uint64
 - data/ids.pkl       : list of ids (strings or ints)
 - data/docs.pkl      : list of original document strings
 - data/shingles.pkl  : optional list of sets (each doc's shingles)

This file is intended to replace the previous numeric-vector generator when
you want to run the MinHash + LSH (Jaccard) pipeline.
"""


# large prime for modular hashing (fits in 64-bit Python int math)
_PRIME = (1 << 61) - 1


def _stable_shingle_hash(sh: str) -> int:
    """Stable integer fingerprint for a shingle (use SHA1 then take 8 bytes)."""
    h = hashlib.sha1(sh.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % _PRIME


class MinHash:
    """Simple MinHash signature generator using linear hashing family."""

    def __init__(self, num_perm: int = 128, seed: int = 42):
        self.num_perm = int(num_perm)
        rng = np.random.RandomState(seed)
        # coefficients a,b for linear family: h_i(x) = (a_i * x + b_i) mod PRIME
        # ensure a_i non-zero
        self.a = rng.randint(1, _PRIME - 1, size=self.num_perm, dtype=np.int64)
        self.b = rng.randint(0, _PRIME - 1, size=self.num_perm, dtype=np.int64)

    def signature(self, shingles: Set[str]) -> np.ndarray:
        """Compute minhash signature for a set of shingles. Returns 1D uint64 array."""
        if not shingles:
            # empty set -> use max val sentinel
            return np.full(self.num_perm, _PRIME, dtype=np.uint64)

        # convert shingles to ints
        sh_ints = np.array([_stable_shingle_hash(s) for s in shingles], dtype=np.int64)
        # compute (a[:,None] * sh_ints[None,:] + b[:,None]) % PRIME => shape (num_perm, n_sh)
        a = self.a.astype(object)[:, None]  # object to avoid overflow in intermediate (but Python handles big ints)
        b = self.b.astype(object)[:, None]
        # we do computation in Python ints via vectorization loop to be robust in modulus
        # For moderate doc sizes this is acceptable; for extreme scale consider optimized C extension.
        sig = np.empty(self.num_perm, dtype=np.uint64)
        for i in range(self.num_perm):
            vals = (int(self.a[i]) * sh_ints + int(self.b[i])) % _PRIME
            sig[i] = int(np.min(vals))
        return sig

    def batch_signature(self, shingles_list: List[Set[str]]) -> np.ndarray:
        """Compute signatures for many documents. Returns array (N, num_perm) dtype uint64."""
        sigs = [self.signature(s) for s in shingles_list]
        return np.vstack(sigs).astype(np.uint64)


# --------------------------
# Shingling helpers
# --------------------------
def shingle_document(doc: str, k: int = 5, by_word: bool = True) -> Set[str]:
    """Return a set of shingles for the document.
    - if by_word=True: k is number of words
    - else: k is number of characters (char-grams)
    """
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


# --------------------------
# Synthetic doc generator
# --------------------------
def make_synthetic_docs(n_docs: int = 10000,
                        vocab_size: int = 1000,
                        avg_words: int = 50,
                        sigma_words: float = 10.0,
                        out_dir: str = "data",
                        seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Create synthetic documents by sampling words from a synthetic vocabulary.

    Returns (docs, ids) and writes docs and ids to out_dir.
    """
    rng = np.random.RandomState(seed)
    os.makedirs(out_dir, exist_ok=True)

    # build a synthetic vocabulary: word0, word1, ...
    vocab = [f"w{idx}" for idx in range(vocab_size)]

    docs = []
    ids = []
    for i in range(n_docs):
        # number of words for this doc (clamp to >=1)
        n_words = max(1, int(rng.normal(loc=avg_words, scale=sigma_words)))
        words = rng.choice(vocab, size=n_words, replace=True)
        doc_text = " ".join(words)
        docs.append(doc_text)
        ids.append(f"doc_{i:06d}")

    # save original docs & ids (pickle)
    with open(os.path.join(out_dir, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)
    with open(os.path.join(out_dir, "ids.pkl"), "wb") as f:
        pickle.dump(ids, f)

    print(f"Saved {n_docs} synthetic docs to {out_dir}/docs.pkl and ids to ids.pkl")
    return docs, ids


# --------------------------
# Build MinHash signatures for a set of documents and save
# --------------------------
def build_and_save_minhash_signatures(docs: List[str],
                                     ids: List[str],
                                     num_perm: int = 128,
                                     k_shingle: int = 3,
                                     by_word: bool = True,
                                     out_dir: str = "data",
                                     save_shingles: bool = True,
                                     seed: int = 42) -> np.ndarray:
    """
    Build MinHash signatures for docs and save to out_dir as sigs.npy (uint64).
    Also saves shingles list if save_shingles True.
    Returns signatures array shape (N, num_perm).
    """
    os.makedirs(out_dir, exist_ok=True)
    # shingle docs
    shingles_list = [shingle_document(d, k=k_shingle, by_word=by_word) for d in docs]

    # build minhash
    mh = MinHash(num_perm=num_perm, seed=seed)
    sigs = mh.batch_signature(shingles_list)  # shape (N, num_perm), dtype uint64

    # save signatures and ancillary info
    np.save(os.path.join(out_dir, "sigs.npy"), sigs)
    with open(os.path.join(out_dir, "ids.pkl"), "wb") as f:
        pickle.dump(ids, f)
    # store MH object params for query time (we only store seed/num_perm/k_shingle/by_word)
    with open(os.path.join(out_dir, "minhash_meta.pkl"), "wb") as f:
        pickle.dump({"num_perm": num_perm, "k_shingle": k_shingle, "by_word": by_word, "seed": seed}, f)

    if save_shingles:
        with open(os.path.join(out_dir, "shingles.pkl"), "wb") as f:
            pickle.dump(shingles_list, f)

    print(f"Saved signatures to {out_dir}/sigs.npy (shape={sigs.shape}), metadata/minhash_meta.pkl")
    return sigs


# --------------------------
# Inspect / quick stats
# --------------------------
def inspect_signatures(sigs: np.ndarray, docs: List[str], ids: List[str], n_sample: int = 5):
    print("Signatures stats:")
    print(f" - shape: {sigs.shape}")
    print(f" - dtype: {sigs.dtype}")
    print(f" - sample rows (first {n_sample}):")
    print(sigs[:n_sample])
    print("\nSample documents:")
    for i in range(min(n_sample, len(docs))):
        print(f" - id={ids[i]} len(doc)={len(docs[i])} -> {docs[i][:120]}...")


# --------------------------
# CLI / script entrypoint
# --------------------------
if __name__ == "__main__":
    # parameters: tweak to your needs
    N_DOCS = 20000
    VOCAB = 20
    AVG_WORDS = 40
    SIGMA_WORDS = 10.0
    OUT_DIR = "data"
    NUM_PERM = 128
    K_SHINGLE = 1
    BY_WORD = True
    SEED = 42

    docs, ids = make_synthetic_docs(n_docs=N_DOCS,
                                    vocab_size=VOCAB,
                                    avg_words=AVG_WORDS,
                                    sigma_words=SIGMA_WORDS,
                                    out_dir=OUT_DIR,
                                    seed=SEED)

    sigs = build_and_save_minhash_signatures(docs, ids,
                                            num_perm=NUM_PERM,
                                            k_shingle=K_SHINGLE,
                                            by_word=BY_WORD,
                                            out_dir=OUT_DIR,
                                            save_shingles=True,
                                            seed=SEED)

    inspect_signatures(sigs, docs, ids, n_sample=5)
