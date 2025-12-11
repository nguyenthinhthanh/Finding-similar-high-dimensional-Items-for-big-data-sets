# Finding Similar High-Dimensional Items for Big Data Sets using LSH
**Mục tiêu:** Triển khai hệ thống tìm kiếm từ gần ngữ nghĩa trên tập dữ liệu embedding lớn, sử dụng SimHash + Locality-Sensitive Hashing (LSH) và xử lý phân tán bằng Dask Cluster.
Dự án minh họa cách xây dựng một kiến trúc tìm kiếm tương tự (Approximate Nearest Neighbor Search - ANN) cho dữ liệu nhiều chiều nhưng vẫn đảm bảo tốc độ khi dataset rất lớn.

---
## Giới thiệu
Trong các bài toán xử lý ngôn ngữ tự nhiên (NLP), việc tìm kiếm các từ gần nghĩa về ngữ nghĩa (semantic similarity) dựa trên vector embedding là nhiệm vụ quan trọng, phục vụ:

- Gợi ý từ đồng nghĩa / gần nghĩa  
- Tìm kiếm mở rộng truy vấn  
- Hệ thống gợi ý và matching từ khóa  
- Phân tích văn bản quy mô lớn  

Tuy nhiên, khi tập embedding lên đến **hàng chục triệu từ**, tìm kiếm tuyến tính trở nên quá chậm. Khi dữ liệu ngày càng lớn (documents, users, embeddings, item-sets…), việc tìm các phần tử giống nhau theo Jaccard / cosine / overlap trở nên rất tốn chi phí.

Để giải quyết bài toán, dự án sử dụng:

- **SimHash** để nén vector D chiều thành chữ ký nhị phân ngắn  
- **LSH** để nhóm các từ giống nhau vào cùng bucket  
- **Dask Distributed** để phân tán dữ liệu thành nhiều shard và mở rộng hệ thống theo số lượng worker  

**LSH** giải quyết vấn đề bằng cách:

- Hash các vector/tập lớn thành các vector chữ ký (signature) ngắn hơn bằng **MinHash**
- Chia chữ ký thành **bands**, và đưa các band vào các **hash buckets**
- Hai items càng giống → càng có xác suất vào cùng bucket

Do đó chỉ cần so sánh **một tập ứng viên nhỏ**, thay vì so toàn bộ N² cặp, LSH giảm đáng kể số phép so sánh và tăng tốc độ khi xử lý datasets lớn.

---

## Cấu trúc thư mục
```test
app/
├── src/
│   ├── query_service.py        # API query (FastAPI)
│   ├── index_builder.py        # Tạo shard + index LSH phân tán
│   ├── worker_task.py          # Task xử lý query trên mỗi worker
│   ├── lsh.py                  # SimHash + LSH trên vector nhị phân
│   └── worker_entrypoint.py    # Entrypoint Dask worker
benchmarks/
│   ├── prepare_data.py         # Download GloVe + Word2Vec
│   ├── build_signatures.py     # Tạo SimHash signature matrix
│   └── synth_data.py           # Benchmark & test pipeline
data/
├── glove.840B.300d.txt
├── word2vec-google-news-300.kv
└── sigs.npy                    # Signature matrix (N × 128)
docker/
├── scheduler/
├── worker/
└── query/
docker-compose.yml
README.md
```
---

## Pipeline xử lý
#### **1. Embedding -> SimHash**
- Mỗi vector **D chiều** được ánh xạ thành **chữ ký nhị phân d-bit**.
- Các từ có ý nghĩa gần nhau ⇒ **chữ ký (signature) gần giống nhau**.

#### **2. LSH Buckets**
- Chia signature thành nhiều **bands** để đưa vào bucket.
- Các từ có signature tương tự sẽ rơi vào **cùng bucket**, giúp:
  - Giảm số lượng so sánh
  - Tăng tốc truy vấn trên dataset lớn

#### **3. Sharding phân tán**
- Dataset được chia thành nhiều **shard** kích thước cố định (ví dụ: *5000 từ / shard*).
- Mỗi worker giữ một phần dataset ⇒ **truy vấn song song** trên toàn cụm.

#### **4. Distributed Query (Dask)**
- Query được broadcast tới **toàn bộ worker**.
- Mỗi worker tìm **ứng viên gần nhất** trong shard cục bộ.
- Scheduler hợp nhất kết quả ⇒ trả về **top-k từ gần nghĩa**.

## Yêu cầu
- Docker & Docker Compose.
- Python 3.9+.
  
---

## Hướng dẫn chạy Dask Cluster với Docker Compose
#### 1. Clone repo
```bash
git https://github.com/nguyenthinhthanh/Finding-similar-high-dimensional-Items-for-big-data-sets
```

#### 2. Chuẩn bị dữ liệu
```bash
python benchmarks/prepare_data.py
python benchmarks/build_signatures.py
```
#### 3. Tạo dữ liệu phân tán
```bash
python app/src/index_builder.py \
    --data data/sigs.npy \
    --out data/shards \
    --shard-size 5000 \
    --inspect
```
#### 4. Build Docker images
```bash
docker compose build
```
#### 5. Khởi động cluster
```bash
docker compose up -d --scale worker=3
```

##### ***Tổng quan các service***

| Service   | Số container | Mô tả                |
|------------|---------------|----------------------|
| scheduler  | 1             | Điều phối công việc  |
| worker     | 3             | Tính toán song song  |
| query      | 1             | API HTTP gọi tới Dask cluster |

#### 6. Gửi truy vấn
API:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"vector":[...], "k":10}'
```

Python client:
```bash
python app_client.py
```

## Đóng góp
Bạn có ý tưởng cải thiện dự án? Hãy mở Pull Request hoặc Issue trên GitHub!

## Giấy phép
Dự án này được tạo ra chỉ nhằm mục đích học tập. Không được sử dụng cho mục đích thương mại.


