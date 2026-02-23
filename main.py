from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import geopandas as gpd
from shapely.geometry import shape

import os
from pathlib import Path
from dotenv import load_dotenv

from app.dike_components.dike_model import DikeModel
from app.dike_components.ground_model import GroundModel

load_dotenv()

# API Key configuration
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable must be set")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key

app = FastAPI(title="Verkenning 2.0 Backend", version="1.0.0")

# CORS configuration - restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "https://portal.wsrl.nl"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    return {"message": "Verkenning 2.0 Backend API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Pydantic models for request/response
class GeoJSONFeature(BaseModel):
    type: str
    geometry: Dict[str, Any]
    properties: Dict[str, Any] = {}

class GeoJSONInput(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]
    crs: Optional[Dict[str, Any]] = None

class VolumeCalculationResult(BaseModel):
    net_volume: float
    excavation_volume: float
    fill_volume: float
    area: float
    unit: str = "m³"
    calculation_time: Optional[float] = None
    grid_points: Optional[int] = None

class RuimtebeslagResult(BaseModel):
    type: str = "FeatureCollection"
    features: List[Dict[str, Any]]
    crs: Optional[Dict[str, Any]] = None
    total_area_m2: float
    num_polygons: Optional[int] = None
    points_above_ground: Optional[int] = None
    calculation_time: Optional[float] = None

class DesignCalculationResult(BaseModel):
    volume: VolumeCalculationResult
    calculation_time: float
    ruimtebeslag_2d_points: List[Any]  # Points data for calculting ruimtebeslag in the frontend

class DesignCostResult(BaseModel):
    breakdown: dict  # Please dont change type, Pydantic is being very annoying

class VolumeCalculationRequest(BaseModel):
    geojson: GeoJSONInput
    excavation_mode: str = 'envelope'  # or 'cut_and_fill'

class CostCalculationRequest(BaseModel):
    geojson_dike: GeoJSONInput | None = None
    geojson_structure: GeoJSONInput | None = None
    complexity: str = "gemiddelde maatregel"
    excavation_mode: str = 'envelope'  # or 'cut_and_fill'
    road_surface: float = 0.0
    number_houses: int = 0


@app.post("/api/calculate_designs", response_model=DesignCalculationResult)
async def calculate_designs(
    request: VolumeCalculationRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Calculates design volume and returns points for 2D ruimtebeslag from GeoJSON input.
    
    Expects a GeoJSON FeatureCollection with 3D polygon features.
    Returns volume calculations and a list of points (EPSG:3857) for ruimtebeslag calculation in the frontend.
    """
    import time
    start_time = time.time()
    
    try:
        if not request.geojson.features:
            raise HTTPException(status_code=400, detail="No features provided in GeoJSON")
        
        # Convert GeoJSON to GeoDataFrame
        features = []
        for feature in request.geojson.features:
            geom = shape(feature.geometry)
            features.append({
                'geometry': geom,
                **feature.properties
            })
        
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        
        # Check if geometries are 3D
        if not gdf.geometry.iloc[0].has_z:
            raise HTTPException(
                status_code=400, 
                detail="Geometry must be 3D (include Z coordinates for elevation)"
            )
        
        # Initialize GroundModel with the GeoDataFrame
        ground_model = GroundModel(
            gdf,
            excavation_mode=request.excavation_mode,
        )
        
        # Calculate volume using Matthias's method
        volume_start = time.time()
        result = ground_model.calculate_volume()
        
        print(f"DEBUG: Result type: {type(result)}")
        print(f"DEBUG: Result value: {result}")
        
        # If result is None, extract from ground_model attributes
        if isinstance(result, dict):
            print(f"DEBUG: Result is dict: {result}")
            fill_vol = result.get('fill_volume', 0.0)
            cut_vol = result.get('cut_volume', 0.0)
            total_vol = result.get('net_volume', 0.0)
            area = result.get('area', 0.0)
            grid_pts = result.get('grid_points', None)
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")
        
        volume_time = time.time() - volume_start
        
        # Calculate 2D ruimtebeslag
        ruimtebeslag_result = ground_model.calculate_ruimtebeslag_2d()
        
        
        total_calculation_time = time.time() - start_time
        
        # Build volume calculation result
        volume_calc = VolumeCalculationResult(
            net_volume=round(total_vol, 2),
            excavation_volume=round(cut_vol, 2),
            fill_volume=round(fill_vol, 2),
            area=round(area, 2),
            calculation_time=round(volume_time, 3),
            grid_points=grid_pts
        )
        
        # Build ruimtebeslag GeoJSON with metadata
        ruimtebeslag_points = ruimtebeslag_result['points_above_ground']
        # ruimtebeslag_2d_points can now be a list as per the updated Pydantic model
        return DesignCalculationResult(
            volume=volume_calc,
            calculation_time=round(total_calculation_time, 3),
            ruimtebeslag_2d_points=ruimtebeslag_points,
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Error calculating designs: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=500, 
            detail=f"Error calculating designs: {str(e.detail) if hasattr(e, 'detail') else str(e)}"
        )


@app.post("/api/cost_calculation", response_model=DesignCostResult)
async def calculate_total_cost(
        payload: CostCalculationRequest,
        api_key: str = Depends(verify_api_key)
):
    """

    """
    print(payload)



    try:
        #soil part
        if not payload.geojson_dike == None:
            features = []
            for feature in payload.geojson_dike.features:
                geom = shape(feature.geometry)
                features.append({'geometry': geom, **feature.properties})

            gdf_ground = gpd.GeoDataFrame(features, crs="EPSG:4326")
        else:
            gdf_ground = None
        
        #structure part
        if not payload.geojson_structure == None:
            features = []
            for feature in payload.geojson_structure.features:
                geom = shape(feature.geometry)
                features.append({'geometry': geom, **feature.properties})

            gdf_structure = gpd.GeoDataFrame(features, crs="EPSG:4326")
        else:
            gdf_structure = None

        dike_model = DikeModel(
            _3d_ground_polygon=gdf_ground,
            _2d_structure=gdf_structure,
            complexity=payload.complexity,
            excavation_mode=payload.excavation_mode,
        )

        cost_breakdown = dike_model.compute_cost(road_area=payload.road_surface,
                                                 nb_houses=payload.number_houses)
        

        print(cost_breakdown)

        return DesignCostResult(
            breakdown=cost_breakdown
        )

    except Exception as e:
        import traceback
        error_detail = f"Error calculating designs: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=500,
            detail=f"Error calculating designs: {str(e)}"
        )


# Example endpoint for design operations
@app.get("/api/designs")
async def get_designs():
    """Get all saved designs"""
    return {"designs": []}

@app.post("/api/designs")
async def create_design(design: dict):
    """Create a new design"""
    return {"message": "Design created", "design": design}


@app.get("/api/datasets")
async def list_datasets(api_key: str = Depends(verify_api_key)):
    """List available CSV files in app/datasets directory"""
    try:
        datasets_path = Path("app/datasets")
        
        if not datasets_path.exists():
            raise HTTPException(status_code=404, detail="Datasets directory not found")
        
        # Get all CSV files
        csv_files = [f.name for f in datasets_path.glob("*.csv")]
        
        return {
            "datasets": csv_files,
            "count": len(csv_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@app.get("/api/datasets/{filename}")
async def download_dataset(filename: str, api_key: str = Depends(verify_api_key)):
    """Download a specific CSV file from app/datasets directory"""
    try:
        # Security: only allow CSV files and prevent directory traversal
        if not filename.endswith(".csv") or ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = Path("app/datasets") / filename
        
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
        
        return FileResponse(
            path=str(file_path),
            media_type="text/csv",
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
