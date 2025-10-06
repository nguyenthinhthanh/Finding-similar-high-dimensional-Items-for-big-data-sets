# Dùng Python 3.13 làm base image
FROM python:3.13

# Đặt thư mục làm việc trong container
WORKDIR /app

# Copy file requirements vào container
COPY app/requirements.txt /app/

# Cài đặt dependencies và dọn dẹp cho nhẹ
RUN apt-get update && apt-get install -y build-essential \
 && pip install --upgrade pip \
 && pip install -r requirements.txt \
 && apt-get remove -y build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy toàn bộ mã nguồn ứng dụng vào container
COPY app /app

# Chạy ứng dụng FastAPI bằng Uvicorn
CMD ["uvicorn", "query_service:app", "--host", "0.0.0.0", "--port", "8000"]
