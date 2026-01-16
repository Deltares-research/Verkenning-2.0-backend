# Digital Ocean App Platform Deployment Guide

## Prerequisites
- Docker installed locally (for testing)
- Git repository connected to Digital Ocean
- Digital Ocean account

## Local Testing

Test the Docker container locally before deploying:

```bash
# Build the Docker image
docker build -t demo-backend .

# Run the container
docker run -p 8000:8000 demo-backend

# Test the API
# Open browser to http://localhost:8000
# Or use curl: curl http://localhost:8000/health
```

## Digital Ocean App Platform Deployment

### Option 1: Using Dockerfile (Recommended)

1. **Push your code to Git** (GitHub, GitLab, or Bitbucket):
   ```bash
   git add Dockerfile .dockerignore .python-version
   git commit -m "Add Docker support for Digital Ocean"
   git push origin main
   ```

2. **Create App on Digital Ocean**:
   - Go to [Digital Ocean App Platform](https://cloud.digitalocean.com/apps)
   - Click "Create App"
   - Connect your Git repository
   - Digital Ocean will auto-detect the Dockerfile

3. **Configure App Settings**:
   - **Name**: demo-backend
   - **Type**: Web Service
   - **Dockerfile Path**: `Dockerfile` (auto-detected)
   - **HTTP Port**: 8000
   - **Health Check Path**: `/health`

4. **Environment Variables** (REQUIRED):
   - Go to App Settings → Environment Variables
   - Add the following variable:
     - **Key**: `API_KEY`
     - **Value**: Your secure API key (generate a strong random string)
     - **Type**: Secret (encrypted)
   - Example: `API_KEY=your-secure-random-key-here-min-32-chars`

5. **Deploy**:
   - Click "Next" through the wizard
   - Review settings
   - Click "Create Resources"

### Option 2: Using App Spec (Advanced)

Create a `.do/app.yaml` file:

```yaml
name: demo-backend
services:
- name: web
  dockerfile_path: Dockerfile
  source_dir: /
  github:
    repo: your-username/demo_backend
    branch: main
    deploy_on_push: true
  http_port: 8000
  health_check:
    http_path: /health
  instance_count: 1
  instance_size_slug: basic-xxs
  routes:
  - path: /
```

Then deploy using doctl CLI:
```bash
doctl apps create --spec .do/app.yaml
```

## Deployment Notes

### Why Docker?
The standard Python buildpack failed because:
- `pyproj` requires the PROJ library (system dependency)
- `rasterio` requires GDAL (system dependency)
- `shapely` requires GEOS (system dependency)

Docker gives full control over system dependencies.

### Port Configuration
- The app runs on port 8000
- Digital Ocean will automatically route external traffic to this port
- No need to change the port in `main.py`

### Performance Considerations
- Start with `basic-xxs` or `basic-xs` instance
- Monitor memory usage (geospatial libraries can be memory-intensive)
- Scale up if needed based on traffic

### Monitoring
- Check logs in Digital Ocean App Platform dashboard
- Health check endpoint: `/health`
- API documentation: `/docs` (FastAPI auto-generated)

## Troubleshooting

### Build fails with "out of memory"
- Increase instance size during build
- Or use multi-stage Docker build (advanced)

### App crashes on startup
- Check logs in Digital Ocean dashboard
- Verify environment variables are set correctly
- Test Docker image locally first

### Slow performance
- Geospatial calculations are CPU-intensive
- Consider upgrading instance size
- Add caching if processing same data repeatedly

## Local Development

```bash
# Install dependencies locally (with PROJ/GDAL installed on your system)
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /api/calculate_design_volume` - Volume calculation (main endpoint)
- `POST /api/debug_calculate_volume` - Debug endpoint
- `GET /docs` - Interactive API documentation
