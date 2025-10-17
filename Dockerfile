# Dockerfile này giả định rằng lệnh 'git lfs pull' đã được chạy TRƯỚC KHI build.
# Hãy sử dụng tính năng "Build Command" của Railway để làm việc này.

FROM python:3.9-slim

WORKDIR /app

# Sao chép file requirements.txt trước để tận dụng Docker layer cache
COPY requirements.txt .

# Tối ưu hóa: Cài đặt build dependencies, cài đặt các thư viện Python,
# và gỡ bỏ build dependencies trong cùng MỘT layer để giảm dung lượng image cuối cùng.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy==1.24.3" "Cython<3.0" && \
    pip install --no-cache-dir --no-build-isolation "scikit-surprise==1.1.3" && \
    pip install --no-cache-dir -r requirements.txt && \
    \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# Sao chép toàn bộ mã nguồn, BAO GỒM CẢ các file lớn đã được Railway tải về
COPY . .

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Lệnh khởi động Gunicorn đã được tối ưu của bạn
CMD ["gunicorn", "-w", "1", "--preload", "--timeout", "300", \
    "--worker-tmp-dir", "/dev/shm", \
    "--max-requests", "100", "--max-requests-jitter", "10", \
    "--bind", "0.0.0.0:8000", "recommendation_api:app"]