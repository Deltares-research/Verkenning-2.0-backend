"""
Test script with new input data.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import geopandas as gpd
from shapely.geometry import shape
from app.dike_model import DikeModel

# New input data
input_geojson = {
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
            [5.605224768153091, 51.895764079821106, 7.5],
            [5.607046480372475, 51.89510601605683, 7.5],
            [5.60786772690824, 51.894623004573546, 7.5],
            [5.607976831307955, 51.89469401159619, 11.2],
            [5.607142234599198, 51.89518487547184, 11.2],
            [5.605304613562746, 51.89584868666671, 11.2],
            [5.605224768153091, 51.895764079821106, 7.5]
          ]
        ]
      },
      "properties": {
        "name": "-30.4m_-19.5m",
        "area_3d": 1596.5098616451694
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.605304613562746, 51.89584868666671, 11.2],
            [5.607142234599198, 51.89518487547184, 11.2],
            [5.607976831307955, 51.89469401159619, 11.2],
            [5.608291133981822, 51.89489856336535, 10.8],
            [5.607418078928262, 51.89541204802368, 10.8],
            [5.605534628625057, 51.89609241616928, 10.8],
            [5.605304613562746, 51.89584868666671, 11.2]
          ]
        ]
      },
      "properties": {
        "name": "-19.5m_11.9m",
        "area_3d": 4009.142985984193
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.605534628625057, 51.89609241616928, 10.8],
            [5.607418078928262, 51.89541204802368, 10.8],
            [5.608291133981822, 51.89489856336535, 10.8],
            [5.608346187292249, 51.89493439240966, 7.5],
            [5.607466395897025, 51.89545183932671, 7.5],
            [5.605574918144323, 51.89613510759537, 7.5],
            [5.605534628625057, 51.89609241616928, 10.8]
          ]
        ]
      },
      "properties": {
        "name": "11.9m_17.4m",
        "area_3d": 235.94413436814
      }
    }
  ]
}

print("Converting input GeoJSON to GeoDataFrame...")
features = []
for feature in input_geojson['features']:
    geom = shape(feature['geometry'])
    features.append({
        'geometry': geom,
        **feature['properties']
    })

gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
print(f"Created GeoDataFrame with {len(gdf)} features")

print("\nInitializing DikeModel...")
dike_model = DikeModel(gdf)

print("\nCalculating volume...")
result = dike_model.calculate_volume()

print("\n=== RESULTS ===")
print(f"Total volume: {result.get('total_volume', 0):.2f} m³")
print(f"2D above ground area: {result.get('above_ground_area', 0):.2f} m²")
print(f"3D above ground area: {result.get('above_ground_area_3d', 0):.2f} m²")

# Calculate expected from properties
expected_3d = 1596.51 + 4009.14 + 235.94
print(f"\nExpected 3D area from properties: {expected_3d:.2f} m²")
print(f"Difference: {result.get('above_ground_area_3d', 0) - expected_3d:.2f} m²")

print("\nSaving polygons to output directory...")
path_2d, path_3d = dike_model.save_above_ground_polygons(output_dir="output", prefix="ruimtebeslag")

print("\n=== CHECKING OUTPUT ===")
with open(path_2d, 'r') as f:
    output_2d = json.load(f)
    num_features = len(output_2d.get('features', []))
    print(f"2D output has {num_features} features")

print("\nDone!")
