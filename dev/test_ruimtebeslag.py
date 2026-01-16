"""
Test script for the 2D ruimtebeslag calculation endpoint.
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# API configuration
API_KEY = os.getenv("API_KEY", "your-api-key-here")
BASE_URL = "http://localhost:8000"

# Sample 3D GeoJSON polygon (WGS84 coordinates with Z values)
# This is a simple square elevated above ground
sample_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [4.3821, 51.9225, 15.0],  # lon, lat, elevation
                    [4.3825, 51.9225, 15.0],
                    [4.3825, 51.9228, 15.5],
                    [4.3821, 51.9228, 15.5],
                    [4.3821, 51.9225, 15.0]
                ]]
            },
            "properties": {
                "name": "Test Design Area",
                "description": "Sample elevated polygon for testing"
            }
        }
    ]
}


def test_ruimtebeslag_calculation():
    """Test the ruimtebeslag calculation endpoint."""
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print("Testing 2D Ruimtebeslag Calculation...")
    print(f"API URL: {BASE_URL}/api/calculate_ruimtebeslag")
    print(f"\nInput GeoJSON:")
    print(json.dumps(sample_geojson, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/calculate_ruimtebeslag",
            headers=headers,
            json=sample_geojson,
            params={"alpha": 5.0}  # Alpha parameter for concave hull
        )
        
        print(f"\n=== Response Status: {response.status_code} ===")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"\nRuimtebeslag Results:")
            print(f"  Total Area: {result.get('total_area_m2', 0):.2f} m²")
            print(f"  Number of Polygons: {result.get('num_polygons', 0)}")
            print(f"  Points Above Ground: {result.get('points_above_ground', 0)}")
            print(f"  Calculation Time: {result.get('calculation_time', 0):.3f}s")
            
            if result.get('features'):
                print(f"\n  Polygon Features:")
                for i, feature in enumerate(result['features']):
                    area = feature['properties'].get('area_m2', 0)
                    print(f"    Polygon {i+1}: {area:.2f} m²")
            
            # Save result to file
            output_file = "output/ruimtebeslag_2d.geojson"
            os.makedirs("output", exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    'type': result['type'],
                    'features': result['features'],
                    'crs': result.get('crs')
                }, f, indent=2)
            print(f"\n  Output saved to: {output_file}")
            
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("Make sure the server is running with: uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def test_with_existing_data():
    """Test with existing ruimtebeslag_3d.geojson if available."""
    
        # 3D output test removed
    
    if not os.path.exists(input_file):
        print(f"\nℹ️  File {input_file} not found, skipping this test.")
        return
    
    print(f"\n\nTesting with existing data from {input_file}...")
    
    with open(input_file, 'r') as f:
        geojson_data = json.load(f)
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/calculate_ruimtebeslag",
            headers=headers,
            json=geojson_data,
            params={"alpha": 3.0}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS with existing data!")
            print(f"  Total Area: {result.get('total_area_m2', 0):.2f} m²")
            print(f"  Number of Polygons: {result.get('num_polygons', 0)}")
            print(f"  Calculation Time: {result.get('calculation_time', 0):.3f}s")
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    print("="*60)
    print("2D RUIMTEBESLAG CALCULATION TEST")
    print("="*60)
    
    # Test with sample data
    test_ruimtebeslag_calculation()
    
    # Test with existing file if available
    test_with_existing_data()
    
    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)
