"""
Test specific volume calculation with provided input.
Expected volume: ~4070 m³
"""
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "your-api-key-here")
BASE_URL = "http://localhost:8000"

# Input data from user
test_geojson = {
  "type": "FeatureCollection",
  "crs": {
    "type": "name",
    "properties": {
      "name": "EPSG:4326"
    }
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.594062739778743, 51.894431735305744, 7.4],
            [5.595409180212468, 51.89420245492566, 7.4],
            [5.596885559195035, 51.89475121458416, 7.4],
            [5.596811492869292, 51.89482749217633, 11.5],
            [5.595387934696768, 51.89429836587567, 11.5],
            [5.594100904741471, 51.89451752912013, 11.5],
            [5.594062739778743, 51.894431735305744, 7.4]
          ]
        ]
      },
      "properties": {
        "name": "-26.3m_-16.4m",
        "area_3d": 1213.309092251679
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.594100904741471, 51.89451752912013, 11.5],
            [5.595387934696768, 51.89429836587567, 11.5],
            [5.596811492869292, 51.89482749217633, 11.5],
            [5.596627448184585, 51.89501703023091, 11.7],
            [5.595335142418126, 51.89453669003279, 11.7],
            [5.594195739522718, 51.89473071369153, 11.7],
            [5.594100904741471, 51.89451752912013, 11.5]
          ]
        ]
      },
      "properties": {
        "name": "-16.4m_8.2m",
        "area_3d": 2189.3953221486363
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.594195739522718, 51.89473071369153, 11.7],
            [5.595335142418126, 51.89453669003279, 11.7],
            [5.596627448184585, 51.89501703023091, 11.7],
            [5.5965698403833075, 51.8950763571229, 6.9],
            [5.595318617891078, 51.8946112874252, 6.9],
            [5.5942254237653035, 51.89479744217855, 6.9],
            [5.594195739522718, 51.89473071369153, 11.7]
          ]
        ]
      },
      "properties": {
        "name": "8.2m_15.9m",
        "area_3d": 783.6177734971092
      }
    }
  ]
}


def test_volume_calculation():
    """Test volume calculation with specific input."""
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print("="*70)
    print("VOLUME CALCULATION TEST")
    print("="*70)
    print(f"\nExpected volume: ~4070 m³")
    print(f"Number of features: {len(test_geojson['features'])}")
    
    for i, feature in enumerate(test_geojson['features']):
        props = feature['properties']
        print(f"\nFeature {i+1}: {props.get('name', 'unnamed')}")
        print(f"  3D Area: {props.get('area_3d', 0):.2f} m²")
        coords = feature['geometry']['coordinates'][0]
        z_values = [c[2] for c in coords]
        print(f"  Z range: {min(z_values):.1f}m - {max(z_values):.1f}m")
    
    try:
        print(f"\n{'='*70}")
        print("Sending request to API...")
        print(f"{'='*70}\n")
        
        response = requests.post(
            f"{BASE_URL}/api/calculate_design_volume",
            headers=headers,
            json=test_geojson,
            timeout=60
        )
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n{'='*70}")
            print("✅ RESULTS")
            print(f"{'='*70}")
            print(f"\n  Fill Volume:       {result.get('fill_volume', 0):>10.2f} m³")
            print(f"  Excavation Volume: {result.get('excavation_volume', 0):>10.2f} m³")
            print(f"  Total Volume:      {result.get('total_volume', 0):>10.2f} m³")
            print(f"  Grid Area:         {result.get('area', 0):>10.2f} m²")
            print(f"  Grid Points:       {result.get('grid_points', 0):>10}")
            print(f"  Calculation Time:  {result.get('calculation_time', 0):>10.3f}s")
            
            expected = 4070
            total_vol = result.get('total_volume', 0)
            diff = abs(total_vol - expected)
            diff_pct = (diff / expected) * 100 if expected > 0 else 0
            
            print(f"\n{'='*70}")
            print("COMPARISON")
            print(f"{'='*70}")
            print(f"\n  Expected:   {expected:>10.2f} m³")
            print(f"  Calculated: {total_vol:>10.2f} m³")
            print(f"  Difference: {diff:>10.2f} m³ ({diff_pct:.1f}%)")
            
            if diff_pct < 5:
                print(f"\n  ✅ Result is within 5% tolerance - GOOD!")
            elif diff_pct < 10:
                print(f"\n  ⚠️  Result is within 10% tolerance - ACCEPTABLE")
            else:
                print(f"\n  ❌ Result differs by more than 10% - NEEDS REVIEW")
            
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("Make sure the server is running with:")
        print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    except requests.exceptions.Timeout:
        print("\n❌ ERROR: Request timed out (>60s)")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def test_ruimtebeslag():
    """Also test ruimtebeslag calculation with this data."""
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print(f"\n\n{'='*70}")
    print("RUIMTEBESLAG 2D CALCULATION TEST")
    print(f"{'='*70}\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/calculate_ruimtebeslag",
            headers=headers,
            json=test_geojson,
            params={"alpha": 5.0},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ RUIMTEBESLAG RESULTS")
            print(f"{'='*70}")
            print(f"\n  Total Area:        {result.get('total_area_m2', 0):>10.2f} m²")
            print(f"  Number of Polygons:{result.get('num_polygons', 0):>10}")
            print(f"  Points Above Ground:{result.get('points_above_ground', 0):>10}")
            print(f"  Calculation Time:  {result.get('calculation_time', 0):>10.3f}s")
            
            # Save output
            output_file = "output/test_ruimtebeslag_output.geojson"
            os.makedirs("output", exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    'type': result['type'],
                    'features': result['features'],
                    'crs': result.get('crs')
                }, f, indent=2)
            print(f"\n  Output saved to: {output_file}")
            
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")


if __name__ == "__main__":
    # Test volume calculation
    test_volume_calculation()
    
    # Test ruimtebeslag calculation
    test_ruimtebeslag()
    
    print(f"\n{'='*70}")
    print("All tests completed!")
    print(f"{'='*70}\n")
