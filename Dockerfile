# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.11

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
ENV PYTHONUNBUFFERED True

# Set model directory environment variables
ENV EASYOCR_MODULE_PATH=/app/models
ENV MODULE_PATH=/app/models

# Copy application dependency manifests to the container image.
COPY requirements.txt ./

# Install production dependencies.
RUN pip install -r requirements.txt

# Create necessary directories and pre-download EasyOCR models
RUN mkdir -p /app/models && \
    python -c "import easyocr; reader = easyocr.Reader(\
        lang_list=['en', 'id'], \
        gpu=False, \
        model_storage_directory='/app/models', \
        download_enabled=True, \
        detector=True, \
        recognizer=True)" && \
    ls -la /app/models

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY src/ ./src/

# Run the web service on container startup.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 src.main:app