"""
Test 3D surface area calculation locally without API
"""
import sys
import time

import shapely

from app.dike_components.dike_model import DikeModel

sys.path.insert(0, '..')

import json
import geopandas as gpd
from shapely.geometry import shape

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
              5.568245382650625,
              51.890881614488016,
              8.8
            ],
            [
              5.571640932264274,
              51.891476523815435,
              8.8
            ],
            [
              5.57160887667709,
              51.89154657512496,
              8.9
            ],
            [
              5.568213322196268,
              51.890951664878685,
              8.9
            ],
            [
              5.568245382650625,
              51.890881614488016,
              8.8
            ]
          ]
        ]
      },
      "properties": {
        "name": "-34.9m_-26.8m"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              5.568213322196268,
              51.890951664878685,
              8.9
            ],
            [
              5.57160887667709,
              51.89154657512496,
              8.9
            ],
            [
              5.571578008239614,
              51.891614031932704,
              10.1
            ],
            [
              5.56818244907187,
              51.891019120801595,
              10.1
            ],
            [
              5.568213322196268,
              51.890951664878685,
              8.9
            ]
          ]
        ]
      },
      "properties": {
        "name": "-26.8m_-19m"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              5.56818244907187,
              51.891019120801595,
              10.1
            ],
            [
              5.571578008239614,
              51.891614031932704,
              10.1
            ],
            [
              5.571537245930476,
              51.89170310949893,
              14
            ],
            [
              5.568141680573571,
              51.891108197199316,
              14
            ],
            [
              5.56818244907187,
              51.891019120801595,
              10.1
            ]
          ]
        ]
      },
      "properties": {
        "name": "-19m_-8.7m"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              5.568141680573571,
              51.891108197199316,
              14
            ],
            [
              5.571537245930476,
              51.89170310949893,
              14
            ],
            [
              5.571508751793368,
              51.89176537730326,
              14
            ],
            [
              5.568113182110042,
              51.891170464186835,
              14
            ],
            [
              5.568141680573571,
              51.891108197199316,
              14
            ]
          ]
        ]
      },
      "properties": {
        "name": "-8.7m_-1.5m"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              5.568113182110042,
              51.891170464186835,
              14
            ],
            [
              5.571508751793368,
              51.89176537730326,
              14
            ],
            [
              5.5714921301769955,
              51.891801700185695,
              12.8
            ],
            [
              5.568096557969922,
              51.891206786592775,
              12.8
            ],
            [
              5.568113182110042,
              51.891170464186835,
              14
            ]
          ]
        ]
      },
      "properties": {
        "name": "-1.5m_2.7m"
      }
    }
  ]
}

print("Converting GeoJSON to GeoDataFrame...")
features = []
for feature in geojson_input['features']:
    geom = shape(feature['geometry'])
    features.append({'geometry': geom, **feature['properties']})

gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
print(f"Created GeoDataFrame with {len(gdf)} features")

print("\nInitializing DikeModel...")
dike_model = DikeModel(gdf)


print("\nCalculating volumes and direct cost...")
costs = dike_model.compute_cost(10, 10)
print(costs)
