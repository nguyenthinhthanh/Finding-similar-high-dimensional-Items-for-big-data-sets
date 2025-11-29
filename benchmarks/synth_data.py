import os, requests, zipfile
import urllib.request

import numpy as np
import pandas as pd
from tqdm import tqdm
import gensim.downloader as api
from gensim.models import KeyedVectors


# ================== CẤU HÌNH CƠ BẢN ==================

DATA_DIR = "data"

GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_ZIP = os.path.join(DATA_DIR, "glove.840B.300d.zip")
GLOVE_TXT = os.path.join(DATA_DIR, "glove.840B.300d.txt")

W2V_LOCAL = os.path.join(DATA_DIR, "word2vec-google-news-300.kv")
W2V_BIN   = os.path.join(DATA_DIR, "word2vec-google-news-300.gz")  # file tải thô

X_PATH = os.path.join(DATA_DIR, "X.npy")                    # data chính (đÃ LỌC, có nghĩa)
META_PATH = os.path.join(DATA_DIR, "meta.parquet")         # map id -> word/source
RAW_PATH = os.path.join(DATA_DIR, "raw.npy")  # data test nhỏ (100000 x 128)
RAW_META_PATH = os.path.join(DATA_DIR, "meta_raw.parquet")

# ================== HÀM PHỤ TRỢ ==================

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_glove():
    if not os.path.isfile(GLOVE_ZIP):
        print(f"Tải {GLOVE_ZIP} từ {GLOVE_URL} ...")
        resp = requests.get(GLOVE_URL, stream=True)
        total_size = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading GloVe", ascii=".=") as pbar:
            with open(GLOVE_ZIP, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print(f"Tải {GLOVE_ZIP} hoàn tất.")


def extract_glove():
    """Giải nén glove.840B.300d.txt vào thư mục data/ nếu chưa có."""
    ensure_data_dir()
    if os.path.isfile(GLOVE_TXT):
        print(f"Đã có {GLOVE_TXT}, bỏ qua bước giải nén.")
        return
    if not os.path.isfile(GLOVE_ZIP):
        raise FileNotFoundError(f"Không tìm thấy {GLOVE_ZIP}, hãy tải GloVe trước.")
    print(f"Giải nén {GLOVE_ZIP} ...")
    with zipfile.ZipFile(GLOVE_ZIP, "r") as z:
        # file bên trong zip tên là "glove.840B.300d.txt"
        z.extract("glove.840B.300d.txt", path=DATA_DIR)
    print(f"Đã giải nén {GLOVE_TXT}.")


def is_clean_word(w: str) -> bool:
    """
    Chỉ giữ token toàn chữ cái (A-Z, a-z) và dài >= 2.
    Bỏ số, dấu câu, token 1 ký tự.
    """
    return w.isalpha() and len(w) >= 2


# ================== HÀM CHÍNH LÀM DATA ==================

def make_data() -> np.ndarray:
    """
    Tạo tập dữ liệu từ GloVe + Word2Vec:
    - Tải / giải nén GloVe nếu cần
    - Đọc GloVe và Word2Vec, LỌC NGAY TỪ ĐẦU:
        chỉ giữ word toàn chữ cái và dài >= 2
    - Ghép lại thành X.npy (N x 300, float32)
    - Tạo meta.parquet (word, source, id)
    - Tạo data/raw.npy: 100000 vector, 128 chiều đầu, sample từ X
    """
    ensure_data_dir()

    # 1. Chuẩn bị GloVe
    download_glove()
    extract_glove()

    words = []
    sources = []
    vectors = []

    # 2. Đọc GloVe (đã LỌC trong lúc đọc)
    print("Đang đọc và lọc vector GloVe (840B, 300d)...")
    # Nếu muốn nhanh hơn có thể bỏ total, ở đây để progress bar đẹp
    with open(GLOVE_TXT, "r", encoding="utf8", errors="ignore") as f:
        for line in tqdm(f, desc="GloVe"):
            parts = line.rstrip().split(" ")
            if len(parts) < 301:
                continue  # dòng lỗi, bỏ
            word = parts[0]
            if not is_clean_word(word):
                continue  # loại số, dấu câu, token 1 ký tự
            try:
                vec = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if vec.shape[0] != 300:
                continue
            words.append(word)
            sources.append("glove")
            vectors.append(vec)

    print(f"GloVe: giữ lại {len(words)} từ sau khi lọc.")

    # 3. Đọc Word2Vec GoogleNews (cũng LỌC)
    if not os.path.isfile(W2V_LOCAL):
        print("Đang tải và load Word2Vec GoogleNews 300d qua Gensim...")
        wv = api.load("word2vec-google-news-300")
        wv.save(W2V_LOCAL)
        print(f"Đã lưu cache W2V vào {W2V_LOCAL}.")
    if os.path.isfile(W2V_LOCAL):
        print(f"Đang load Word2Vec từ cache local: {W2V_LOCAL} ...")
        wv = KeyedVectors.load(W2V_LOCAL, mmap='r')
        print("Load xong Word2Vec từ cache.")

    print("Đang lọc và lấy vector Word2Vec...")
    existing = set(words)
    count_before = len(words)
    added = 0

    for word in tqdm(wv.index_to_key, desc="Word2Vec"):
        if not is_clean_word(word):
            continue
        if word in existing:
            continue  # từ này GloVe đã có rồi, bỏ qua
    
        vec = wv[word]
        if vec.shape[0] != 300:
            continue
        words.append(word)
        sources.append("w2v")
        vectors.append(vec.astype(np.float32))
        added += 1

    print(f"Word2Vec: thêm {added} từ sau khi lọc.")
    total_count = len(words)
    print(f"Tổng số từ SAU LỌC: {total_count}")

    # 4. Ghép thành ma trận X (N x 300, float32)
    print("Đang ghép các vector lại thành ma trận X...")
    X = np.vstack(vectors).astype(np.float32)
    print(f"X shape = {X.shape}")

    # 5. Tạo meta (id trùng index) và lưu
    print("Đang tạo và lưu meta.parquet...")
    df_meta = pd.DataFrame({
        "word": words,
        "source": sources,
        "id": np.arange(total_count, dtype=np.int32),
    })
    df_meta.to_parquet(META_PATH, index=False)
    print(f"Đã lưu meta.parquet (tổng từ = {len(df_meta)}).")

    # 6. Lưu X.npy (đÃ LỌC, chỉ còn từ có nghĩa)
    np.save(X_PATH, X)
    print(f"Đã lưu X.npy với shape = {X.shape}.")

    # 7. Tạo tập nhỏ để test: data/raw.npy (100000 x 128)
    if X.shape[0] >= 100_000:
        print("Đang tạo data/raw.npy (100000 x 128) để test...")
        np.random.seed(42)
        idx = np.random.choice(X.shape[0], size=100_000, replace=False)
        Y = X[idx, :128]  # 128 chiều đầu

        os.makedirs(DATA_DIR, exist_ok=True)
        np.save(RAW_PATH, Y.astype(np.float32))
        print(f"Đã lưu {RAW_PATH} với shape = {Y.shape}.")

        # === META NHỎ TƯƠNG ỨNG VỚI raw.npy ===
        # Lấy các dòng gốc trong meta tương ứng với idx
        df_meta_raw = df_meta.iloc[idx].copy()

        # Lưu id gốc nếu muốn (tùy bạn giữ hay bỏ)
        df_meta_raw["orig_id"] = df_meta_raw["id"]

        # TẠO id MỚI 0..len(raw)-1, TƯƠNG ỨNG raw[i]
        df_meta_raw["id"] = np.arange(len(df_meta_raw), dtype=np.int32)

        # Reset index DataFrame cho gọn
        df_meta_raw = df_meta_raw.reset_index(drop=True)

        df_meta_raw.to_parquet(RAW_META_PATH, index=False)
        print(f"Đã lưu meta_raw tại {RAW_META_PATH} (số dòng = {len(df_meta_raw)}).")
    else:
        print("WARNING: Số từ < 100000, bỏ qua bước tạo raw.npy.")

    return X


# ================== HÀM INSPECT ==================

def inspect_data(X: np.ndarray):
    """In một số thống kê cơ bản và vài dòng đầu của X."""
    print("\n=== DATA INSPECTION ===")
    print(f"- Shape: {X.shape}")
    print(f"- Mean: {X.mean():.6f}")
    print(f"- Std:  {X.std():.6f}")
    print(f"- Min:  {X.min():.6f}")
    print(f"- Max:  {X.max():.6f}")

    print("\n5 vector đầu tiên:")
    print(X[:5])

def get_word(word_id: int, meta_id: int = 1) -> str:
    """
    Trả về 'word' ứng với id cho trước.
    meta_id = 1  -> dùng meta.parquet
    meta_id = 2  -> dùng meta_raw.parquet
    """
    if meta_id == 2:
        meta_path = RAW_META_PATH
    else:
        meta_path = META_PATH

    df = pd.read_parquet(meta_path)

    # tìm đúng dòng có id = word_id
    row = df[df["id"] == word_id]
    if row.empty:
        raise ValueError(f"id {word_id} không tồn tại trong {meta_path}")

    row = row.iloc[0]
    # chỉ trả về word (string)
    return str(row["word"])

def find_word(meta_id: int, word: str) -> int:
    """
    Từ 1 word -> trả về id tương ứng trong meta.
    meta_id = 1 -> dùng META_PATH
    meta_id = 2 -> dùng RAW_META_PATH
    Nếu có nhiều dòng trùng word, trả về id của dòng đầu tiên.
    """
    if meta_id == 2:
        meta_path = RAW_META_PATH
    else:
        meta_path = META_PATH

    df = pd.read_parquet(meta_path)
    rows = df[df["word"] == word]

    if rows.empty:
        raise ValueError(f"'{word}' không có trong tập {meta_path}.")

    # lấy dòng đầu tiên
    row = rows.iloc[0]
    return int(row["id"])


# ================== MAIN ==================

if __name__ == "__main__":
    X = make_data()
    # inspect_data("data/X.npy")
