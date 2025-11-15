# Finding Similar High-Dimensional Items for Big Data Sets using LSH
**Mục tiêu:** Triển khai hệ thống tìm kiếm các mục tương tự (similar items) trong dữ liệu nhiều chiều bằng kỹ thuật Locality-Sensitive Hashing (LSH).
Dự án minh họa cách áp dụng MinHash + LSH để xử lý datasets lớn, giảm số lượng so sánh và tăng hiệu năng tìm kiếm gần đúng (Approximate Nearest Neighbor Search). Repo này cung cấp code demo, pipeline chạy bằng Docker + Docker Compose, và các notebook/benchmark dùng Dask để scale lên nhiều worker

---
## Giới thiệu

Khi dữ liệu ngày càng lớn (documents, users, embeddings, item-sets…), việc tìm các phần tử giống nhau theo Jaccard / cosine / overlap trở nên rất tốn chi phí.

**LSH** giải quyết vấn đề bằng cách:

- Hash các vector/tập lớn thành các vector chữ ký (signature) ngắn hơn bằng **MinHash**
- Chia chữ ký thành **bands**, và đưa các band vào các **hash buckets**
- Hai items càng giống → càng có xác suất vào cùng bucket

Do đó chỉ cần so sánh **một tập ứng viên nhỏ**, thay vì so toàn bộ N² cặp, LSH giảm đáng kể số phép so sánh và tăng tốc độ khi xử lý datasets lớn.

---

## Cấu trúc thư mục
- `docs/` - tóm tắt thuật toán, tham khảo, slides demo.
- `docker/` - Dockerfile(s) cho service (api, worker, indexer), mẫu `docker-compose.yml`.
- `app/src` - mã nguồn Python:
  - `index_builder.py` - logic xây dựng chỉ mục phân tán (sharding, partitioning).
  - `minhash_lsh.py` - MinHash + LSH (banding) implementation for Jaccard similarity on documents.
  - `query_service.py` - Dask-based Query Service.
  - `worker_entrypoint.py` - Dask worker entrypoint.
  - `worker_task.py` - Worker-side Tasks for Dask-based Query Service.
- `benchmarks/` - scripts chạy benchmark (throughput, latency, recall).
- `data/` - scripts tải / sinh dataset thử nghiệm.
    - `shards/` - chứa dữ liệu phân tán sau khi sharding.
- `README.md`.

---

## Tóm tắt phương pháp tiếp cận
Phương pháp chính là sử dụng LSH (Locality-Sensitive Hashing): mỗi mục được chuyển thành một signature ngắn bằng MinHash, sau đó được hash vào các bucket sao cho các mục tương tự có xác suất cao rơi vào cùng bucket. Khi truy vấn, chỉ cần so sánh các mục trong cùng bucket thay vì toàn bộ dataset, kết hợp với index phân tán nếu dữ liệu lớn để chia tải và tận dụng memory/CPU của nhiều node.

---

## Yêu cầu
- Docker & Docker Compose.
- Python 3.9+.  

---

## Hướng dẫn chạy Dask Cluster với Docker Compose
#### 1. Clone repo
```bash
git https://github.com/nguyenthinhthanh/Finding-similar-high-dimensional-Items-for-big-data-sets
```

#### 2. Build images
```bash
# build
docker compose build

# Khởi động cluster
docker compose up -d --scale worker=3
```
##### Tổng quan các service

| Service   | Số container | Mô tả                |
|------------|---------------|----------------------|
| scheduler  | 1             | Điều phối công việc  |
| worker     | 3             | Tính toán song song  |
| query      | 1             | API HTTP gọi tới Dask cluster |

### 3. Kiểm tra log
```bash
docker compose logs --tail=200 scheduler
docker compose logs --tail=200 worker
docker compose logs --tail=200 query
```
### 4. Kiểm tra Dask Dashboard
```bash
Mặc định, Dask Dashboard sẽ được expose tại: http://localhost:8787
```
### 5. Gửi truy vấn
```bash
Lệnh (client → server):
	curl -X POST http://localhost:8000/query \
	  -H "Content-Type: application/json" \
	  -d '{"vector":[...], "k":10}'
```
## Đóng góp
Bạn có ý tưởng cải thiện dự án? Hãy mở Pull Request hoặc Issue trên GitHub!

## Giấy phép
Dự án này được tạo ra chỉ nhằm mục đích học tập. Không được sử dụng cho mục đích thương mại.


