# Verkenning 2.0 Backend

FastAPI backend for Verkenning 2.0 application with volume calculations for dike designs.

## Features

- 🔐 API Key authentication
- 🌍 CORS protection with whitelisted domains
- 📊 3D volume calculations for dike models
- 🗺️ GeoJSON input support
- 📚 Interactive API documentation with custom branding

## Prerequisites

- Python 3.12+
- PROJ, GDAL, and GEOS libraries (for geospatial operations)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/vincentwolf89/demo_backend.git
cd demo_backend
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the root directory:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux/Mac
cp .env.example .env
```

Edit the `.env` file and set your API key:

```env
# API Key for authentication (REQUIRED)
API_KEY=your-secure-api-key-here

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

**Generate a secure API key:**

```powershell
# PowerShell
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})
```

```bash
# Linux/Mac
openssl rand -base64 32
```

## Running the Server

### Development Mode

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reload on code changes.

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once running, visit:
- **API docs**: http://localhost:8000/docs (Interactive Swagger UI with logo)
- **Health check**: http://localhost:8000/health

## Using the API

### Authentication

All protected endpoints require an API key in the request header:

```bash
curl -X POST http://localhost:8000/api/calculate_design_volume \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d @geojson_input.json
```

### JavaScript Example

```javascript
fetch('http://localhost:8000/api/calculate_design_volume', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-api-key-here'
  },
  body: JSON.stringify({
    type: "FeatureCollection",
    features: [/* your GeoJSON features */]
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

### CORS Configuration

The API only accepts requests from:
- `http://localhost:3001` (local development)
- `https://portal.wsrl.nl` (production)

To add more origins, edit `main.py`:

```python
allow_origins=[
    "http://localhost:3001",
    "https://portal.wsrl.nl",
    "https://your-new-domain.com"  # Add your domain here
]
```

## Testing

```bash
# Install pytest
pip install pytest httpx

# Run tests
pytest
```

## Deployment

For deployment to Digital Ocean App Platform with Docker, see [DEPLOYMENT.md](DEPLOYMENT.md).

### Environment Variables for Production

When deploying to Digital Ocean or other cloud platforms:

1. **Never commit `.env` to Git** - it's already in `.gitignore`
2. Set environment variables in your hosting platform:
   - **Digital Ocean**: App Settings → Environment Variables
   - **Heroku**: Settings → Config Vars
   - **AWS**: Systems Manager → Parameter Store
3. Mark `API_KEY` as **Secret/Encrypted** for security

### Security Notes

- ✅ Change the default API key before deploying to production
- ✅ Use a strong, randomly generated API key (32+ characters)
- ✅ Store API keys securely (never in code or public repos)
- ✅ Rotate API keys periodically
- ✅ Update `allow_origins` to restrict CORS to your actual domains

## Project Structure

```
demo_backend/
├── main.py                 # FastAPI application and endpoints
├── volume_calc.py          # Dike volume calculation logic
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in git)
├── .env.example            # Environment variables template
├── Dockerfile              # Docker configuration
├── .dockerignore           # Docker build exclusions
├── .python-version         # Python version specification
├── static/                 # Static files (logos, etc.)
│   └── deltares_logo.png
├── .do/                    # Digital Ocean configuration
│   └── app.yaml
└── README.md               # This file
```

## API Endpoints

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/` | GET | No | API information |
| `/health` | GET | No | Health check |
| `/docs` | GET | No | Interactive API documentation |
| `/api/calculate_design_volume` | POST | Yes | Calculate dike volume from GeoJSON |
| `/api/debug_calculate_volume` | POST | Yes | Debug volume calculation |

## Troubleshooting

### "API_KEY environment variable must be set"
- Make sure you've created the `.env` file
- Verify `API_KEY` is set in the `.env` file
- Check the `.env` file encoding (should be UTF-8 without BOM)

### "Invalid API Key" (403 error)
- Verify you're sending the correct API key in the `X-API-Key` header
- Check that the API key matches the one in your `.env` file

### CORS errors
- Ensure your frontend domain is listed in `allow_origins` in `main.py`
- Check that you're sending the `X-API-Key` header correctly

### Import errors with geospatial libraries
- Windows: Install prebuilt wheels from [Christoph Gohlke's repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/)
- Linux: Install system packages: `apt-get install libgdal-dev libproj-dev libgeos-dev`
- Mac: Use Homebrew: `brew install gdal proj geos`

## License

[Add your license here]

## Contact

For questions or issues, please contact the development team or open an issue on GitHub.
