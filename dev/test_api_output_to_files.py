# Test the /api/calculate_designs endpoint with the provided input

# Test the /api/calculate_designs endpoint with the provided input
import json
import requests
import os
from dotenv import load_dotenv

# Set the alpha parameter for ruimtebeslag calculation
ALPHA = 1.0  # Change this value for a tighter or looser fit

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable not set! (Check your .env file)")
import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
  raise RuntimeError("API_KEY environment variable not set! (Check your .env file)")

input_geojson = {
  "type": "FeatureCollection",
  "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.598428002888296, 51.895254716780414, 8],
            [5.599760601536858, 51.89548320931427, 8],
            [5.599721801664576, 51.89556982725364, 12.1],
            [5.598389200645032, 51.89534133428359, 12.1],
            [5.598428002888296, 51.895254716780414, 8]
          ]
        ]
      },
      "properties": {"name": "-23.5m_-13.5m", "area_3d": 654.1088579980826}
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.598389200645032, 51.89534133428359, 12.1],
            [5.599721801664576, 51.89556982725364, 12.1],
            [5.599634501406232, 51.89576471756704, 12],
            [5.598301895051944, 51.8955362236155, 12],
            [5.598389200645032, 51.89534133428359, 12.1]
          ]
        ]
      },
      "properties": {"name": "-13.5m_9m", "area_3d": 896.0240295543426}
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.598301895051944, 51.8955362236155, 12],
            [5.599634501406232, 51.89576471756704, 12],
            [5.5996127732245, 51.895813223589855, 7.3],
            [5.5982801655424455, 51.895584729393995, 7.3],
            [5.598301895051944, 51.8955362236155, 12]
          ]
        ]
      },
      "properties": {"name": "9m_14.6m", "area_3d": 334.2621529215663}
    }
  ]
}


url = f"http://localhost:8000/api/calculate_designs?alpha={ALPHA}"
headers = {
  "accept": "application/json",
  "Content-Type": "application/json",
  "X-API-Key": API_KEY
}

response = requests.post(url, headers=headers, data=json.dumps({"features": input_geojson["features"]}))

if response.status_code == 200:
    result = response.json()
    # # Save 2D ruimtebeslag
    # with open("output/ruimtebeslag_2d.geojson", "w", encoding="utf-8") as f2d:
    #     json.dump(result["ruimtebeslag_2d"], f2d, indent=2)
    # # Save 3D ruimtebeslag
    # # 3D output removed
    print("Saved 2D and 3D ruimtebeslag outputs to output/ directory.")
    print(result["ruimtebeslag_2d_points"])
else:
    print(f"API call failed: {response.status_code}\n{response.text}")
