# benchmarks/prepare_data.py
import os
import requests
import zipfile
import sys
from tqdm import tqdm

# --- CHECK LIBRARIES ---
try:
    import gensim.downloader as api
except ImportError:
    print("[PREPARE] Error: Thiếu thư viện 'gensim'. Hãy chạy: pip install gensim")
    # Không exit, vì có thể chỉ cần GloVe
    
# --- CẤU HÌNH ---
DATA_DIR = "data"

# URLs & Paths
GLOVE_URL = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
GLOVE_ZIP = os.path.join(DATA_DIR, "glove.840B.300d.zip")
GLOVE_TXT = os.path.join(DATA_DIR, "glove.840B.300d.txt")

W2V_LOCAL = os.path.join(DATA_DIR, "word2vec-google-news-300.kv")

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def download_glove():
    ensure_data_dir()
    
    # 1. Tải file ZIP nếu chưa có
    if not os.path.exists(GLOVE_ZIP) and not os.path.exists(GLOVE_TXT):
        print(f"[PREPARE] Đang tải GloVe 840B (Khoảng 2GB)...")
        try:
            resp = requests.get(GLOVE_URL, stream=True)
            total_size = int(resp.headers.get("Content-Length", 0))
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Download GloVe") as pbar:
                with open(GLOVE_ZIP, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024*1024): # Chunk 1MB
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            print("[PREPARE] Tải xong GloVe ZIP.")
        except Exception as e:
            print(f"[PREPARE] Lỗi tải GloVe: {e}")
            sys.exit(1)

    # 2. Giải nén nếu chưa có file TXT
    if not os.path.exists(GLOVE_TXT):
        if os.path.exists(GLOVE_ZIP):
            print(f"[PREPARE] Đang giải nén GloVe...")
            try:
                with zipfile.ZipFile(GLOVE_ZIP, "r") as z:
                    z.extract("glove.840B.300d.txt", path=DATA_DIR)
                print("[PREPARE] Giải nén xong: glove.840B.300d.txt")
            except zipfile.BadZipFile:
                print("[PREPARE] Lỗi: File Zip bị hỏng. Hãy xóa đi tải lại.")
        else:
            print("[PREPARE] Không tìm thấy file Zip để giải nén.")
    else:
        print("[PREPARE] Đã có sẵn file GloVe txt.")

def download_word2vec():
    ensure_data_dir()
    
    if os.path.exists(W2V_LOCAL):
        print("[PREPARE] Đã có sẵn file Word2Vec.")
        return

    print("[PREPARE] Đang tải Word2Vec (Google News) qua Gensim...")
    try:
        # Gensim tự tải về cache, ta load lên rồi save lại vào folder data để synth_data dùng
        model = api.load("word2vec-google-news-300")
        print(f"[PREPARE] Đang lưu model xuống {W2V_LOCAL}...")
        model.save(W2V_LOCAL)
        print("[PREPARE] Lưu xong Word2Vec.")
    except Exception as e:
        print(f"[PREPARE] Cảnh báo: Không tải được Word2Vec ({e}). Hệ thống sẽ chỉ dùng GloVe.")

if __name__ == "__main__":
    print("=== CHUẨN BỊ NGUYÊN LIỆU THÔ (RAW DATA) ===")
    
    # 1. Chuẩn bị GloVe (Bắt buộc)
    download_glove()
    
    # 2. Chuẩn bị Word2Vec (Tùy chọn, để bổ sung vốn từ)
    download_word2vec()
    
    print("\n[SUCCESS] Hoàn tất chuẩn bị dữ liệu!")
    print("👉 Bước tiếp theo: Chạy 'python benchmarks/synth_data.py' để lọc, gộp và tạo vector SimHash.")