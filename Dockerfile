# Use the official Python image with slim variant
FROM python:3.11-slim

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED=True \
    APP_HOME=/app \
    EASYOCR_MODULE_PATH=/app/models \
    MODULE_PATH=/app/models

WORKDIR $APP_HOME

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /app/models

# Pre-download EasyOCR models in a separate layer
RUN python -c "import easyocr; reader = easyocr.Reader(['en', 'id'], \
    gpu=False, \
    model_storage_directory='/app/models', \
    download_enabled=True, \
    detector=True, \
    recognizer=True)"

# Copy local code to the container image
COPY src/ ./src/

# Run the web service on container startup
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 src.main:app