# app_client.py
import pandas as pd
import numpy as np
import requests
import os
import sys

# CẤU HÌNH
DATA_DIR = "data"
META_PATH = os.path.join(DATA_DIR, "meta.parquet") # Từ điển
VEC_PATH = os.path.join(DATA_DIR, "X.npy")         # Kho Vector
QUERY_URL = "http://localhost:8000/query"          # Địa chỉ Server

class Vocabulary:
    def __init__(self, meta_path):
        print(f"Loading vocabulary from {meta_path}...")
        self.df = pd.read_parquet(meta_path)
        # Tạo map cho nhanh
        self.word_to_id = dict(zip(self.df['word'], self.df['id']))
        self.id_to_word = dict(zip(self.df['id'], self.df['word']))
        print(f"Loaded {len(self.df)} words.")

    def get_id(self, word):
        return self.word_to_id.get(word, None)

    def get_word(self, idx):
        return self.id_to_word.get(idx, "<Unknown>")

class VectorStore:
    def __init__(self, vec_path):
        print(f"Loading vectors from {vec_path}...")
        # Dùng mmap_mode='r' để không load hết vào RAM nếu file lớn
        self.data = np.load(vec_path, mmap_mode='r')
        print(f"Vector shape: {self.data.shape}")

    def get_vector(self, idx):
        if 0 <= idx < self.data.shape[0]:
            return self.data[idx] # Trả về numpy array
        return None

def main():
    # 1. Khởi tạo
    if not os.path.exists(META_PATH) or not os.path.exists(VEC_PATH):
        print("LỖI: Không tìm thấy file data/meta.parquet hoặc data/X.npy")
        print("Hãy chạy 'python benchmarks/synth_data.py' trước!")
        return

    vocab = Vocabulary(META_PATH)
    store = VectorStore(VEC_PATH)

    print("\n--- SYSTEM READY (Type 'exit' to quit) ---")
    
    while True:
        # 2. Input Word
        user_input = input("\nNhập từ khóa (Word): ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # 3. Word -> ID
        word_id = vocab.get_id(user_input)
        if word_id is None:
            print(f"Không tìm thấy từ '{user_input}' trong từ điển.")
            continue
            
        print(f"-> Found ID: {word_id}")
        
        # 4. ID -> Vector
        vector = store.get_vector(word_id)
        if vector is None:
            print("Lỗi: ID có trong từ điển nhưng không tìm thấy vector.")
            continue
            
        # 5. Send Vector to Server
        try:
            # Convert vector float32 -> list để gửi JSON
            payload = {
                "vector": vector.tolist(), 
                "k": 10
            }
            print(f"-> Sending query to {QUERY_URL}...")
            response = requests.post(QUERY_URL, json=payload)
            
            if response.status_code == 200:
                results = response.json().get("candidates", [])
                print(f"\nKẾT QUẢ TÌM KIẾM CHO '{user_input}':")
                print("-" * 50)
                print(f"{'RANK':<5} {'WORD':<20} {'SIMILARITY (Score)':<20} {'ID':<10}")
                print("-" * 50)
                
                for i, res in enumerate(results):
                    # Server trả về ID dạng [shard_idx, row_idx]. 
                    # Nếu bạn chưa mapping lại global ID ở server, ta cần tính lại:
                    # Giả sử shard_size = 5000 (cần khớp với config server)
                    # Hoặc nếu server trả về global ID thì dùng luôn.
                    # Ở đây tôi giả định server trả về (shard_idx, row_idx)
                    
                    shard_idx, row_idx = res['id']
                    # CÔNG THỨC GLOBAL ID: (Cần cấu hình SHARD_SIZE đúng với lúc build index)
                    SHARD_SIZE = 100000 # Mặc định trong index_builder.py
                    global_id = shard_idx * SHARD_SIZE + row_idx
                    
                    found_word = vocab.get_word(global_id)
                    score = res['score']
                    
                    print(f"#{i+1:<4} {found_word:<20} {score:.4f}               {global_id}")
            else:
                print(f"Server Error: {response.text}")
                
        except Exception as e:
            print(f"Connection Error: {e}")
            print("Đảm bảo Server (query_service) đang chạy!")

if __name__ == "__main__":
    main()