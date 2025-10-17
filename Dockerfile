# Giai đoạn 1: Builder - Tải các file lớn từ Git LFS
FROM python:3.9-slim as builder

# Cài đặt các gói hệ thống cần thiết: git và git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sao chép toàn bộ project (quan trọng: bao gồm cả thư mục .git)
COPY . .

# Chạy lệnh git lfs để tải về các file thật
# Nếu lệnh này thất bại, quá trình build sẽ dừng lại
RUN git lfs install && git lfs pull

# ---

# Giai đoạn 2: Final Image - Xây dựng môi trường chạy ứng dụng
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

# Sao chép mã nguồn và DỮ LIỆU ĐÃ ĐƯỢC TẢI VỀ từ giai đoạn builder
COPY --from=builder /app .

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Lệnh khởi động Gunicorn đã được tối ưu của bạn
# Sửa lại thành "recommendation_api:app" để khớp với tên file python
CMD ["gunicorn", "-w", "1", "--preload", "--timeout", "300", \
    "--worker-tmp-dir", "/dev/shm", \
    "--max-requests", "100", "--max-requests-jitter", "10", \
    "--bind", "0.0.0.0:8000", "recommendation_api:app"]