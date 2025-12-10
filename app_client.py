import requests
import time
import sys
import os

# --- CẤU HÌNH ---
API_URL = "http://127.0.0.1:8000/search_text"
TOP_K = 10  # Số lượng kết quả muốn hiển thị

def clear_screen():
    # Hàm xóa màn hình cho gọn (Windows dùng cls, Linux dùng clear)
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 60)
    print("      HỆ THỐNG TÌM KIẾM TƯƠNG ĐỒNG (DISTRIBUTED SEARCH)")
    print("=" * 60)
    print(" * Chế độ: Word-to-Word")
    print(" * Backend: Dask Cluster (1 Scheduler, Workers)")
    print(" * Dữ liệu: GloVe + Word2Vec")
    print("-" * 60)
    print(" 👉 Nhập từ tiếng Anh để tìm kiếm (ví dụ: machine, love...)")
    print(" 👉 Nhập 'exit' hoặc 'quit' để thoát chương trình.")
    print("=" * 60)

def search(keyword):
    payload = {"text": keyword, "k": TOP_K}
    start_time = time.time()
    
    try:
        # Gửi request (Timeout 60s phòng trường hợp Worker tính lâu)
        response = requests.post(API_URL, json=payload, timeout=60)
        end_time = time.time()
        duration = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            print(f"\n✅ Kết quả cho '{keyword}' (Thời gian: {duration:.4f}s):")
            print("-" * 50)
            print(f"{'#':<4} {'WORD (TỪ TÌM ĐƯỢC)':<25} {'ĐỘ TƯƠNG ĐỒNG':<15}")
            print("-" * 50)
            
            if not results:
                print("   (Không tìm thấy từ nào tương đồng)")
            
            for i, item in enumerate(results):
                word = item.get('word', 'N/A')
                score = item.get('score', 0.0)
                # In màu giả lập (dùng ký tự ASCII nếu cần) hoặc in thường
                print(f"{i+1:<4} {word:<25} {score:.4f}")
            print("-" * 50)
            
        else:
            print(f"\n❌ Lỗi Server (Code {response.status_code}): {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ LỖI: Không thể kết nối tới Server!")
        print("   👉 Bạn đã chạy file 'run_system.bat' chưa?")
    except requests.exceptions.Timeout:
        print("\n⏳ LỖI: Hết thời gian chờ (Timeout). Worker đang bận.")
    except Exception as e:
        print(f"\n❌ Lỗi không xác định: {e}")

def main():
    clear_screen()
    print_header()
    
    while True:
        try:
            # Nhập liệu từ người dùng
            user_input = input("\n[User] Nhập từ khóa: ").strip()
            
            # Kiểm tra thoát
            if user_input.lower() in ['exit', 'quit']:
                print("\nĐang thoát hệ thống... Tạm biệt! 👋")
                break
            
            # Bỏ qua nếu nhập rỗng
            if not user_input:
                continue
                
            # Thực hiện tìm kiếm
            search(user_input)
            
        except KeyboardInterrupt:
            # Bắt phím Ctrl+C
            print("\n\nĐã ngắt chương trình. Tạm biệt!")
            break

if __name__ == "__main__":
    main()