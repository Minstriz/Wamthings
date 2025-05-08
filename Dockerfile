
# Sử dụng image Python chính thức làm base
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các file cần thiết vào container
COPY app_final.py .
COPY requirements.txt .
COPY templates/ templates/
COPY static/ static/
COPY faces/ faces/
COPY weights/ weights/
COPY report_excel.py .

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện Python từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng 5000 để Railway truy cập
EXPOSE 5000

# Thiết lập biến môi trường cho Flask
ENV FLASK_APP=app_final.py
ENV FLASK_ENV=production

# Lệnh để chạy ứng dụng
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app_final:app"]
