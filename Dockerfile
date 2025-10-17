# Sử dụng Python 3.9 slim để có một image nhẹ
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# 1. Cài đặt các system dependencies, bao gồm gcc/g++ để build, và git/git-lfs
# git-lfs RẤT QUAN TRỌNG để tải các file model và data lớn từ Git LFS
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy file dependencies
COPY requirements.txt /app/

# 3. Cài đặt dependencies build phức tạp (scikit-surprise) trong một layer riêng
# Giúp tối ưu caching và đảm bảo scikit-surprise được build đúng cách
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy==1.24.3" "Cython<3.0" && \
    pip install --no-cache-dir --no-build-isolation "scikit-surprise==1.1.3"

# 4. Cài đặt các dependencies còn lại từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code và data (bao gồm model.pkl và data.csv)
COPY . /app/

# Thiết lập biến môi trường để output log ngay lập tức
ENV PYTHONUNBUFFERED=1

# Cổng mặc định
EXPOSE 8000

# Khởi chạy Gunicorn với các thiết lập tối ưu cho Railway
# -w 1 (1 worker) tốt cho ML model lớn (dùng --preload)
# --worker-tmp-dir /dev/shm sử dụng shared memory cho worker (tối ưu IO)
CMD ["gunicorn", "-w", "1", "--preload", "--timeout", "300", \
    "--worker-tmp-dir", "/dev/shm", \
    "--max-requests", "100", "--max-requests-jitter", "10", \
    "--bind", "0.0.0.0:8000", "app:app"]