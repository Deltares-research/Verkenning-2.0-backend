# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    proj-bin \
    proj-data \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV PROJ_DIR=/usr

# Python runtime safety / memory behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONMALLOC=malloc

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (DO App Platform maps $PORT automatically)
EXPOSE 8000

# Run the application with Gunicorn + Uvicorn worker
# This prevents slow memory growth by recycling workers
CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "1", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--bind", "0.0.0.0:8000"]
