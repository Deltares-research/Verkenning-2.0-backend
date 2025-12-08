# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for geospatial libraries
# PROJ is required for pyproj
# GDAL is required for rasterio and geopandas
# GEOS is required for shapely
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

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Run the application with uvicorn
# Use 0.0.0.0 to accept connections from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
