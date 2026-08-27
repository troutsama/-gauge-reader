FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir onnxruntime  # CPU-only

# App
COPY . .

# Models should be mounted or copied separately
# docker run -v /path/to/models:/app/models ...

EXPOSE 8000

CMD ["python", "app_v2.py", "--host", "0.0.0.0", "--port", "8000"]
