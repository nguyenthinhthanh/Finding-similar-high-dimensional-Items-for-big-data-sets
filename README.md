# Finding similar high-dimensional items using Query-driven Dynamic Quantization + Distributed Indexing (Dask + Docker)

**Mục tiêu:** triển khai, demo và nghiên cứu phương pháp *query-driven dynamic quantization* kết hợp *distributed indexing* để tìm các phần tử tương tự trong không gian nhiều chiều lớn, dựa trên ý tưởng trong Guzun et al. (2019). Repo này cung cấp code demo, pipeline chạy bằng Docker + Docker Compose, và các notebook/benchmark dùng Dask để scale lên nhiều worker

---

## Cấu trúc thư mục
- `docs/` — tóm tắt thuật toán, tham khảo, slides demo.
- `docker/` — Dockerfile(s) cho service (api, worker, indexer), mẫu `docker-compose.yml`.
- `app/` — mã nguồn Python:
  - `indexing/` — logic xây dựng chỉ mục phân tán (sharding, partitioning).
  - `quantization/` — cài đặt Query-driven Dynamic Quantization (QED-like).
  - `query/` — pipeline query: candidate generation → local scoring → global aggregation.
  - `benchmarks/` — scripts chạy benchmark (throughput, latency, recall).
- `data/` — scripts tải / sinh dataset thử nghiệm.
- `README.md`.

---

## Tóm tắt phương pháp tiếp cận
Phương pháp chính là thực hiện **quantization phụ thuộc vào query** (mỗi query sinh ra một hoặc nhiều quantization "cục bộ" để lọc bỏ các điểm không liên quan trước khi tính toán khoảng cách đầy đủ), kết hợp với **index phân tán** để chia tải và tận dụng memory/CPU nhiều node. Kết quả mong đợi: giảm thời gian truy vấn và vẫn giữ recall cao so với scan tuần tự khi không gian chiều cao.

---

## Yêu cầu
- Docker & Docker Compose.
- Python 3.9+.  

---

## Hướng dẫn chạy Dask Cluster với Docker Compose
### 1. Clone repo
```bash
git https://github.com/nguyenthinhthanh/Finding-similar-high-dimensional-Items-for-big-data-sets
```

### 2. Build images
```bash
# build
docker compose build

# Khởi động cluster
docker compose up -d --scale worker=3
```
#### Tổng quan các service

| Service   | Số container | Mô tả                |
|------------|---------------|----------------------|
| scheduler  | 1             | Điều phối công việc  |
| worker     | 3             | Tính toán song song  |
| query      | 1             | API HTTP gọi tới Dask cluster |

### 3. Kiểm tra log
```bash
docker compose logs -f scheduler
docker compose logs -f worker
docker compose logs -f query
```
### 4. Kiểm tra Dask Dashboard
```bash
Mặc định, Dask Dashboard sẽ được expose tại: http://localhost:8787
```
## Đóng góp
- Bạn có ý tưởng cải thiện dự án? Hãy mở Pull Request hoặc Issue trên GitHub!

## Giấy phép
- Null.



