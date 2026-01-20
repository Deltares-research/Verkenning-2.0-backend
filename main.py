from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import geopandas as gpd
from shapely.geometry import shape

import os
from dotenv import load_dotenv

from app.dike_model import DikeModel

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
    total_volume: float
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


@app.post("/api/calculate_designs", response_model=DesignCalculationResult)
async def calculate_designs(
    geojson: GeoJSONInput,
    api_key: str = Depends(verify_api_key)
):
    """
    Calculate design volume and return points for 2D ruimtebeslag from GeoJSON input.
    
    Expects a GeoJSON FeatureCollection with 3D polygon features.
    Returns volume calculations and a list of points (EPSG:3857) for ruimtebeslag calculation in the frontend.
    """
    import time
    start_time = time.time()
    
    try:
        if not geojson.features:
            raise HTTPException(status_code=400, detail="No features provided in GeoJSON")
        
        # Convert GeoJSON to GeoDataFrame
        features = []
        for feature in geojson.features:
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
        
        # Initialize DikeModel with the GeoDataFrame
        dike_model = DikeModel(gdf)
        
        # Calculate volume using Matthias's method
        volume_start = time.time()
        result = dike_model.calculate_volume()
        
        print(f"DEBUG: Result type: {type(result)}")
        print(f"DEBUG: Result value: {result}")
        
        # If result is None, extract from dike_model attributes
        if result is None:
            print("DEBUG: Result is None, extracting from model attributes")
            fill_vol = getattr(dike_model, 'total_fill_volume', 0.0)
            cut_vol = getattr(dike_model, 'total_cut_volume', 0.0)
            total_vol = fill_vol - cut_vol
            area = getattr(dike_model, 'total_area', 0.0)
            grid_pts = getattr(dike_model, 'grid_points_count', None)
            print(f"DEBUG: Extracted - fill: {fill_vol}, cut: {cut_vol}, total: {total_vol}")
        # Handle both tuple and dict return types
        elif isinstance(result, tuple):
            print(f"DEBUG: Result is tuple with length {len(result)}")
            if len(result) >= 3:
                fill_vol = result[0] if len(result) > 0 else 0.0
                cut_vol = result[1] if len(result) > 1 else 0.0
                total_vol = result[2] if len(result) > 2 else 0.0
                area = result[3] if len(result) > 3 else 0.0
                grid_pts = result[4] if len(result) > 4 else None
            else:
                fill_vol = cut_vol = total_vol = area = 0.0
                grid_pts = None
        elif isinstance(result, dict):
            print(f"DEBUG: Result is dict: {result}")
            fill_vol = result.get('fill_volume', 0.0)
            cut_vol = result.get('cut_volume', 0.0)
            total_vol = result.get('total_volume', 0.0)
            area = result.get('area', 0.0)
            grid_pts = result.get('grid_points', None)
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")
        
        volume_time = time.time() - volume_start
        
        # Calculate 2D ruimtebeslag
        ruimtebeslag_result = dike_model.calculate_ruimtebeslag_2d()
        
        
        total_calculation_time = time.time() - start_time
        
        # Build volume calculation result
        volume_calc = VolumeCalculationResult(
            total_volume=round(total_vol, 2),
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
            detail=f"Error calculating designs: {str(e)}"
        )

# Debug endpoint to see raw calculation result
@app.post("/api/debug_calculate_volume")
async def debug_calculate_volume(
    geojson: GeoJSONInput,
    api_key: str = Depends(verify_api_key)
):
    """Debug endpoint to see raw calculation result"""
    try:
        features = []
        for feature in geojson.features:
            geom = shape(feature.geometry)
            features.append({'geometry': geom, **feature.properties})
        
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        dike_model = DikeModel(gdf)
        result = dike_model.calculate_volume()
        
        return {
            "result_type": str(type(result)),
            "result": str(result),
            "is_tuple": isinstance(result, tuple),
            "is_dict": isinstance(result, dict),
            "length": len(result) if isinstance(result, (tuple, list)) else None
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/api/cost_calculation", response_model=DesignCostResult)
async def calculate_total_cost(
        geojson: GeoJSONInput,
        complexity: str,
        road_surface: float,
        number_houses: int,
        api_key: str = Depends(verify_api_key)
):
    """

    """


    try:
        features = []
        for feature in geojson.features:
            geom = shape(feature.geometry)
            features.append({'geometry': geom, **feature.properties})

        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        print(1111111111111)
        dike_model = DikeModel(gdf)

        cost_breakdown = dike_model.compute_cost(road_area=road_surface,
                                                 complexity=complexity,
                                                 nb_houses=number_houses)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
