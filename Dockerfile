# Sử dụng Python 3.9 slim
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# 1. Cài đặt các system dependencies, bao gồm gcc/g++ và git/git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy toàn bộ code (bao gồm các file pointer LFS)
# Bước này phải nằm trước khi cài đặt dependencies để có thể sử dụng LFS
COPY . /app/

# 💡 FIX LFS: Buộc Git LFS tải các file lớn từ Remote trong quá trình build
# Các file model (*.pkl) và data (*.csv) sẽ được tải xuống thực tế tại đây
RUN git lfs pull

# 3. Copy file dependencies (vẫn cần, mặc dù đã copy . ở trên)
# Đây là cách để tận dụng caching: Docker sẽ kiểm tra xem requirements.txt đã thay đổi chưa
COPY requirements.txt /app/

# 4. Cài đặt dependencies build phức tạp (scikit-surprise)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy==1.24.3" "Cython<3.0" && \
    pip install --no-cache-dir --no-build-isolation "scikit-surprise==1.1.3"

# 5. Cài đặt các dependencies còn lại
RUN pip install --no-cache-dir -r requirements.txt

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1

# Cổng mặc định
EXPOSE 8000

# Khởi chạy Gunicorn
CMD ["gunicorn", "-w", "1", "--preload", "--timeout", "300", \
    "--worker-tmp-dir", "/dev/shm", \
    "--max-requests", "100", "--max-requests-jitter", "10", \
    "--bind", "0.0.0.0:8000", "app:app"]