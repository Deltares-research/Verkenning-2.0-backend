"""
Test the /api/calculate_designs endpoint with sample GeoJSON input
"""
import requests
import json

# API endpoint
url = "http://localhost:8000/api/calculate_designs"

# API key from .env
api_key = "GB0joLKaSshreEigJmUqYb9fdnF64RxC"

# Test GeoJSON input
geojson_input = {
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
            [
              5.617396762604224,
              51.892118707255655,
              7.4
            ],
            [
              5.618030071386272,
              51.89194291737966,
              7.4
            ],
            [
              5.619055890942459,
              51.89176166265669,
              7.4
            ],
            [
              5.619095383176657,
              51.89184722581094,
              11.5
            ],
            [
              5.618079504462289,
              51.892026724137885,
              11.5
            ],
            [
              5.617455628963844,
              51.89219989560896,
              11.5
            ],
            [
              5.617396762604224,
              51.892118707255655,
              7.4
            ]
          ]
        ]
      },
      "properties": {
        "name": "-26.3m_-16.4m",
        "area_3d": 955.9611571564992
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              5.617455628963844,
              51.89219989560896,
              11.5
            ],
            [
              5.618079504462289,
              51.892026724137885,
              11.5
            ],
            [
              5.619095383176657,
              51.89184722581094,
              11.5
            ],
            [
              5.619208674865487,
              51.89209267962859,
              11.5
            ],
            [
              5.6182213135018735,
              51.89226713936357,
              11.5
            ],
            [
              5.617624499297118,
              51.89243279940282,
              11.5
            ],
            [
              5.617455628963844,
              51.89219989560896,
              11.5
            ]
          ]
        ]
      },
      "properties": {
        "name": "-16.4m_12m",
        "area_3d": 1655.8372661678995
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              5.617624499297118,
              51.89243279940282,
              11.5
            ],
            [
              5.6182213135018735,
              51.89226713936357,
              11.5
            ],
            [
              5.619208674865487,
              51.89209267962859,
              11.5
            ],
            [
              5.619224232623703,
              51.89212638630563,
              6.9
            ],
            [
              5.618240787396972,
              51.8923001541164,
              6.9
            ],
            [
              5.6176476893739595,
              51.892464782650706,
              6.9
            ],
            [
              5.617624499297118,
              51.89243279940282,
              11.5
            ]
          ]
        ]
      },
      "properties": {
        "name": "12m_15.9m",
        "area_3d": 216.17800362857318
      }
    }
  ]
}

# Make the request
print("Sending request to:", url)
print("API Key:", api_key[:10] + "...")

headers = {
    "x-api-key": api_key,
    "Content-Type": "application/json"
}

params = {
    "alpha": 5.0
}

response = requests.post(url, json=geojson_input, headers=headers, params=params)

print(f"\nStatus Code: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print("\n=== RESULTS ===")
    print(f"\n--- Volume Calculation ---")
    print(f"Total Volume: {result['volume']['total_volume']:.2f} m³")
    print(f"Fill Volume: {result['volume']['fill_volume']:.2f} m³")
    print(f"Excavation Volume: {result['volume']['excavation_volume']:.2f} m³")
    print(f"Area: {result['volume']['area']:.2f} m²")
    print(f"Grid Points: {result['volume']['grid_points']}")
    print(f"Calculation Time: {result['volume']['calculation_time']:.3f}s")
    
    print(f"\n--- Ruimtebeslag (2D Footprint) ---")

    
    print(f"\n--- GeoJSON Output ---")
    print(f"Number of Polygons: {len(result['ruimtebeslag_2d']['features'])}")
    print(f"Total Calculation Time: {result['calculation_time']:.3f}s")
    
    # Save full result to file
    with open('output/api_test_result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\nFull result saved to: output/api_test_result.json")
    
else:
    print(f"\nError Response:")
    print(response.text)
