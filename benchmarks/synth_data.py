# benchmarks/synth_data.py
import os, requests, zipfile
import numpy as np
import gensim.downloader as api
from tqdm import tqdm

# URLs và tên file
def make_data():
    GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
    GLOVE_ZIP = "data/glove.840B.300d.zip"
    GLOVE_TXT = "data/glove.840B.300d.txt"
    # 1. Download GloVe zip nếu chưa có
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
    # 2. Giải nén glove.840B.300d.txt nếu chưa có
    if not os.path.isfile(GLOVE_TXT):
        print(f"Giải nén {GLOVE_ZIP} ...")
        with zipfile.ZipFile(GLOVE_ZIP, "r") as z:
            z.extract("glove.840B.300d.txt", path="data/")
        print(f"Đã giải nén glove.840B.300d.txt.")

    # 3. Load Word2Vec Google News qua gensim
    print("Tải và load Word2Vec Google News 300d qua Gensim...")
    model = api.load("word2vec-google-news-300")
    print("Đã load Word2Vec Google News (300d).")

    # 4. Đếm số từ trong GloVe và Word2Vec
    print("Đếm số từ trong GloVe...")
    glove_count = 0
    with open(GLOVE_TXT, "r", encoding="utf-8") as f:
        for _ in f:
            glove_count += 1
    print(f"Số từ GloVe: {glove_count}")

    w2v_count = len(model.index_to_key)
    print(f"Số từ Word2Vec: {w2v_count}")

    total_count = glove_count + w2v_count
    dim = 300

    # 5. Cấp phát mảng numpy và list meta
    X = np.empty((total_count, dim), dtype=np.float32)
    words = [None] * total_count
    sources = [None] * total_count

    # 6. Đọc GloVe và ghi vào X
    print("Đang đọc và xử lý các vector GloVe...")
    with open(GLOVE_TXT, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=glove_count, desc="Processing GloVe", ascii=".=")):
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue  # bỏ qua dòng lỗi nếu có
            words[i] = parts[0]
            sources[i] = "glove"
            X[i, :] = np.fromiter(parts[1:], dtype=np.float32)

    # 7. Ghi vector Word2Vec vào X
    print("Đang kết hợp các vector Word2Vec vào mảng...")
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

    # 8. Lưu X.npy
    np.save("data/X.npy", X)
    print(f"Đã lưu mảng X.npy (shape={X.shape}).")

    # 9. Tạo raw.npy để test
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], size=100_000, replace=False)
    Y = X[indices]
    np.save('data/raw.npy', Y)
    print(f"Đã lưu mảng raw.npy (shape={Y.shape}).")
    # # 10. Tạo meta.parquet
    # df_meta = pd.DataFrame({"word": words, "source": sources, "id": np.arange(total_count, dtype=np.int32)})
    # df_meta.to_parquet("meta.parquet", index=False)
    # print(f"Đã lưu bảng meta.parquet (tổng từ = {len(df_meta)}).")

    # # 11. Tính stats và lưu stats.json
    # stats = {
    #     "n_samples": int(total_count),
    #     "n_dims": int(dim),
    #     "min": float(X.min()),
    #     "max": float(X.max()),
    #     "mean": float(X.mean()),
    #     "std": float(X.std())
    # }
    # glove_norms = np.linalg.norm(X[:glove_count, :], axis=1)
    # w2v_norms = np.linalg.norm(X[glove_count:glove_count+w2v_count, :], axis=1)
    # stats["avg_L2_norm_glove"] = float(glove_norms.mean())
    # stats["avg_L2_norm_word2vec"] = float(w2v_norms.mean())
    # # Lựa chọn một vài mẫu vector để lưu
    # sample_indices = [0, glove_count-1, glove_count, total_count-1]
    # stats["samples"] = [
    #     {"word": words[idx], "source": sources[idx], "vector": X[idx].tolist()}
    #     for idx in sample_indices
    # ]
    # with open("stats.json", "w", encoding="utf-8") as f:
    #     json.dump(stats, f, ensure_ascii=False, indent=2)
    # print("Đã lưu thống kê stats.json.")
    return X


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

if __name__ == '__main__':
    X = make_data()
    # inspect_data(X)