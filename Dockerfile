# Sử dụng image Python chính thức làm base
FROM python:3.9-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép các file cần thiết vào container
COPY app_final.py /app/app_final.py
COPY requirements.txt /app/requirements.txt
COPY templates/ /app/templates/
COPY static/ /app/static/
COPY faces/ /app/faces/
COPY weights/ /app/weights/
COPY report_excel.py /app/report_excel.py
COPY models/ /app/models/ 
COPY utils/ /app/utils/ 
# Kiểm tra sự tồn tại của app_final.py
RUN ls -l /app/app_final.py || (echo "Error: app_final.py not found" && exit 1)

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện Python từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Kiểm tra xem các thư viện cần thiết đã được cài đặt
RUN pip show gunicorn || (echo "Error: gunicorn not installed" && exit 1)
RUN pip show eventlet || (echo "Error: eventlet not installed" && exit 1)
RUN pip show requests || (echo "Error: requests not installed" && exit 1)
RUN pip show urllib3 || (echo "Error: urllib3 not installed" && exit 1)

# Mở cổng 5000 để Railway truy cập
EXPOSE 5000

# Thiết lập biến môi trường cho Flask
ENV FLASK_APP=app_final
ENV FLASK_ENV=production

# Lệnh để chạy ứng dụng
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "app_final:app"]