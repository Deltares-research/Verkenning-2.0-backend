import sys
from pathlib import Path

# Add parent directory to path to allow 'backend' imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import geopandas as gpd
from shapely.geometry import shape
import json
import time

from volume_calc import DikeModel  # Changed from backend.volume_calc

app = FastAPI(title="Verkenning 2.0 Backend", version="1.0.0")

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.post("/api/calculate_design_volume", response_model=VolumeCalculationResult)
async def calculate_design_volume(geojson: GeoJSONInput):
    """
    Calculate design volume from GeoJSON input using DikeModel.
    
    Expects a GeoJSON FeatureCollection with 3D polygon features.
    Returns volume calculations including excavation and fill volumes.
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
        result = dike_model.calculate_volume_matthias()
        
        print(f"DEBUG: Result type: {type(result)}")
        print(f"DEBUG: Result value: {result}")
        
        calculation_time = time.time() - start_time
        
        # If result is None, extract from dike_model attributes
        if result is None:
            print("DEBUG: Result is None, extracting from model attributes")
            # Access the model's stored values directly
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
        
        return VolumeCalculationResult(
            total_volume=round(total_vol, 2),
            excavation_volume=round(cut_vol, 2),
            fill_volume=round(fill_vol, 2),
            area=round(area, 2),
            calculation_time=round(calculation_time, 3),
            grid_points=grid_pts
        )
        
    except Exception as e:
        import traceback
        error_detail = f"Error calculating volume: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=500, 
            detail=f"Error calculating volume: {str(e)}"
        )

# Debug endpoint to see raw calculation result
@app.post("/api/debug_calculate_volume")
async def debug_calculate_volume(geojson: GeoJSONInput):
    """Debug endpoint to see raw calculation result"""
    try:
        features = []
        for feature in geojson.features:
            geom = shape(feature.geometry)
            features.append({'geometry': geom, **feature.properties})
        
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        dike_model = DikeModel(gdf)
        result = dike_model.calculate_volume_matthias()
        
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
