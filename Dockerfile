# Sử dụng image Python làm base
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy tất cả file từ context local (bao gồm file LFS nếu đã pull)
COPY . /app

# Cài đặt dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng 8000 (hoặc cổng từ biến môi trường)
EXPOSE $PORT

# Chạy ứng dụng
CMD ["python", "app.py"]