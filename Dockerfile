FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# 1. Cài đặt Dependencies Hệ thống BẮT BUỘC cho biên dịch
# Bổ sung build-essential (chứa gcc, g++, make, v.v.) và python3-dev (chứa Python headers).
# Đây là bước quan trọng nhất để khắc phục lỗi 'gcc failed'.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Sao chép và Cài đặt Dependencies Python
COPY requirements.txt /app/

# Chiến lược cài đặt pip tối ưu cho scikit-surprise và các gói biên dịch:
# - Cài đặt pip mới nhất.
# - Cài đặt các gói nền tảng (numpy, Cython) trước, sau đó là scikit-surprise với cờ --no-build-isolation.
# - Cài đặt phần còn lại từ requirements.txt.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy==1.24.3" "Cython<3.0" && \
    pip install --no-cache-dir --no-build-isolation "scikit-surprise==1.1.3" && \
    pip install --no-cache-dir -r requirements.txt

# 3. Sao chép mã nguồn ứng dụng
COPY . /app/

# 4. Cấu hình Môi trường
# Bật Python unbuffered để log xuất hiện ngay lập tức
ENV PYTHONUNBUFFERED=1

# Cổng ứng dụng Gunicorn
EXPOSE 8000

# 5. Lệnh Khởi chạy (Gunicorn)
# Giữ nguyên cấu hình tối ưu cho nền tảng hosting như Railway/Heroku
CMD ["gunicorn", "-w", "1", "--preload", "--timeout", "300", \
    "--worker-tmp-dir", "/dev/shm", \
    "--max-requests", "100", "--max-requests-jitter", "10", \
    "--bind", "0.0.0.0:8000", "app:app"]