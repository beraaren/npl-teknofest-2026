FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıkları (OpenCV + PyAV + YOLO için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Backend requirements — build context bera/ kökünden gelir
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Proje kaynak kodları
COPY src/        /app/src/
COPY Kanal_B/    /app/Kanal_B/
COPY backend/    /app/backend/
COPY config.yaml /app/config.yaml

# Paylaşılan frame dizini
RUN mkdir -p /app/data/frames

# PYTHONPATH: /app altındaki src, Kanal_B, backend Python'dan bulunabilsin
ENV PYTHONPATH=/app

CMD ["uvicorn", "--help"]
