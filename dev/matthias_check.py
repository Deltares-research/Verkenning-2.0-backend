"""
Test 3D surface area calculation locally without API
"""
import sys
import time

import shapely

sys.path.insert(0, '..')

import json
import geopandas as gpd
from shapely.geometry import shape
from app.volume_calc import DikeModel

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

print("\nCalculating volumes...")
# dike_model.calculate_volume_matthias()

print("\nCalculating volumes and direct cost...")
costs = dike_model.compute_cost()
print(costs)

# dike_model.plot_existing_and_new_surface_plotly()

# print("\nCalculating 2D ruimtebeslag...")
# t1 = time.time()
# ruimtebeslag_2d = dike_model.calculate_ruimtebeslag_2d(alpha=5.0)
# t2 = time.time()
# print(f"2D ruimtebeslag calculation took {t2 - t1} seconds")
# # nb_points = len(above_ground_points.get('ruimtebeslag_2d_points', []))
# # area = 0.525 * 0.525 * nb_points
# # print(area)
# gdf = gpd.GeoDataFrame.from_features(ruimtebeslag_2d['features'], crs="EPSG:4326")
# gdf.to_file("ruimtebeslag_2d_PYTHON.geojson", driver="GeoJSON")
#
#
# stop
#
#
# print(f"2D Area: area m²")
# t3 = time.time()
# d0_area = dike_model.calculate_total_3d_surface_area()
# t4 = time.time()
# print(f"3D surface area calculation took {t4 - t3} seconds")
# print(f"3D Surface Area: {d0_area} m²")
# t5 = time.time()
# d_area = dike_model.calculate_3d_surface_area_above_ahn()
# t6 = time.time()
# print(f"3D surface area above AHN calculation took {t6 - t5} seconds")
# # d_area = dike_model.calculate_3d_surface_area_matt()
# print(f"3D Surface Area (Matt method): {d_area} m²")
#
# # save geojson output for inspection
# gdf = gpd.GeoDataFrame.from_features(ruimtebeslag_result['features'], crs="EPSG:4326")
# gdf.to_file("ruimtebeslag_2d_PYTHON.geojson", driver="GeoJSON")

# print("\nConverting ruimtebeslag polygons to RD coordinates...")
# from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon
# from pyproj import Transformer
#
# transformer_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
#
# ruimtebeslag_polygons_rd = []
# for feature in ruimtebeslag_result.get('features', []):
#     geom_wgs = shapely_shape(feature['geometry'])
#     print(f"  Feature geometry type: {type(geom_wgs).__name__}")
#
#     # Handle both Polygon and MultiPolygon
#     if isinstance(geom_wgs, MultiPolygon):
#         print(f"    MultiPolygon with {len(geom_wgs.geoms)} polygons")
#         for poly in geom_wgs.geoms:
#             coords_wgs = list(poly.exterior.coords)
#             coords_rd = [transformer_to_rd.transform(x, y) for x, y in coords_wgs]
#             poly_rd = Polygon(coords_rd)
#             ruimtebeslag_polygons_rd.append(poly_rd)
#     else:
#         # Single Polygon
#         coords_wgs = list(geom_wgs.exterior.coords)
#         coords_rd = [transformer_to_rd.transform(x, y) for x, y in coords_wgs]
#         poly_rd = Polygon(coords_rd)
#         ruimtebeslag_polygons_rd.append(poly_rd)
#
# print(f"\nTotal RD polygons: {len(ruimtebeslag_polygons_rd)}")
#
# print("\nCalculating 3D surface area...")
# try:
#     surface_3d_area = dike_model.calculate_3d_surface_area(ruimtebeslag_polygons_rd)
#     surface_3d_area_matt = dike_model.calculate_3d_surface_area_matt(ruimtebeslag_polygons_rd)
#     print(f"\n=== RESULTS ===")
#     print(f"2D Area: {ruimtebeslag_result.get('total_area_m2', 0):.2f} m²")
#     print(f"3D Surface Area: {surface_3d_area:.2f} m²")
#     print(f"3D Surface Area (Matt method): {surface_3d_area_matt:.2f} m²")
#     print(f"Difference (3D - 2D): {surface_3d_area - ruimtebeslag_result.get('total_area_m2', 0):.2f} m²")
#     print(f"Percentage Increase: {((surface_3d_area / ruimtebeslag_result.get('total_area_m2', 1)) - 1) * 100:.2f}%")
# except Exception as e:
#     print(f"\nERROR: {e}")
#     import traceback
#
#     traceback.print_exc()
