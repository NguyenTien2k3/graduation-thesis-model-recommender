# ===== GIAI ĐOẠN 1: BUILDER - TẢI FILE LFS =====
# Stage này chỉ dùng để tải các file lớn từ Git LFS
FROM alpine/git:latest AS builder

WORKDIR /app

# Sao chép toàn bộ repo vào môi trường build
COPY . .

# ===== GIAI ĐOẠN 2: FINAL IMAGE - XÂY DỰNG ỨNG DỤNG =====
# Stage này sẽ tạo ra image cuối cùng để chạy
FROM python:3.9

WORKDIR /app

# Sao chép file requirements.txt trước
COPY requirements.txt .

# Tối ưu hóa: Cài và gỡ build dependencies trong cùng MỘT layer
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

# Quan trọng: Sao chép các file đã được tải bởi LFS từ stage 'builder'
COPY --from=builder /app .

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1

