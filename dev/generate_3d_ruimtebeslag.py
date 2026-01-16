"""
Generate 3D ruimtebeslag GeoJSON with Z-coordinates
"""
import sys
sys.path.insert(0, '..')

import json
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from app.volume_calc import DikeModel
from matplotlib.path import Path
from scipy.interpolate import griddata
from pyproj import Transformer

# Test GeoJSON input
geojson_input = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.617396762604224, 51.892118707255655, 7.4],
            [5.618030071386272, 51.89194291737966, 7.4],
            [5.619055890942459, 51.89176166265669, 7.4],
            [5.619095383176657, 51.89184722581094, 11.5],
            [5.618079504462289, 51.892026724137885, 11.5],
            [5.617455628963844, 51.89219989560896, 11.5],
            [5.617396762604224, 51.892118707255655, 7.4]
          ]
        ]
      },
      "properties": {"name": "-26.3m_-16.4m"}
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.617455628963844, 51.89219989560896, 11.5],
            [5.618079504462289, 51.892026724137885, 11.5],
            [5.619095383176657, 51.89184722581094, 11.5],
            [5.619208674865487, 51.89209267962859, 11.5],
            [5.6182213135018735, 51.89226713936357, 11.5],
            [5.617624499297118, 51.89243279940282, 11.5],
            [5.617455628963844, 51.89219989560896, 11.5]
          ]
        ]
      },
      "properties": {"name": "-16.4m_12m"}
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.617624499297118, 51.89243279940282, 11.5],
            [5.6182213135018735, 51.89226713936357, 11.5],
            [5.619208674865487, 51.89209267962859, 11.5],
            [5.619224232623703, 51.89212638630563, 6.9],
            [5.618240787396972, 51.8923001541164, 6.9],
            [5.6176476893739595, 51.892464782650706, 6.9],
            [5.617624499297118, 51.89243279940282, 11.5]
          ]
        ]
      },
      "properties": {"name": "12m_15.9m"}
    }
  ]
}

print("Converting GeoJSON to GeoDataFrame...")
features = []
for feature in geojson_input['features']:
    geom = shape(feature['geometry'])
    features.append({'geometry': geom, **feature['properties']})

gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")

print("\nInitializing DikeModel...")
dike_model = DikeModel(gdf)

print("\nCalculating 2D ruimtebeslag...")
ruimtebeslag_result = dike_model.calculate_ruimtebeslag_2d(alpha=5.0)

print("\nGenerating 3D ruimtebeslag GeoJSON with Z-coordinates...")
from shapely.geometry import shape as shapely_shape, Polygon, MultiPolygon
from shapely.ops import unary_union

transformer_to_rd = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
transformer_to_wgs = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

# Get ruimtebeslag polygons in RD
ruimtebeslag_polygons_rd = []
for feature in ruimtebeslag_result.get('features', []):
    geom_wgs = shapely_shape(feature['geometry'])
    
    if isinstance(geom_wgs, MultiPolygon):
        for poly in geom_wgs.geoms:
            coords_wgs = list(poly.exterior.coords)
            coords_rd = [transformer_to_rd.transform(x, y) for x, y in coords_wgs]
            poly_rd = Polygon(coords_rd)
            ruimtebeslag_polygons_rd.append(poly_rd)
    else:
        coords_wgs = list(geom_wgs.exterior.coords)
        coords_rd = [transformer_to_rd.transform(x, y) for x, y in coords_wgs]
        poly_rd = Polygon(coords_rd)
        ruimtebeslag_polygons_rd.append(poly_rd)

# Generate 3D version of ruimtebeslag polygons
features_3d = []

for poly_idx, poly_rd in enumerate(ruimtebeslag_polygons_rd):
    # Get coordinates of the polygon boundary
    coords_rd = np.array(poly_rd.exterior.coords)
    
    # Collect all design polygon coordinates to interpolate from
    all_design_coords = []
    for idx, row in dike_model.design_export_3d.iterrows():
        design_poly = row.geometry
        poly_coords_3d = np.array(design_poly.exterior.coords)
        all_design_coords.append(poly_coords_3d)
    
    # Combine all design coordinates
    all_coords = np.vstack(all_design_coords)
    
    # For each point on the boundary, interpolate Z from all design polygons
    coords_3d = []
    
    for x_rd, y_rd in coords_rd:
        # Interpolate Z from all design coordinates
        z_interp = griddata(
            points=all_coords[:, :2],
            values=all_coords[:, 2],
            xi=np.array([[x_rd, y_rd]]),
            method='linear',
            fill_value=np.nan
        )
        
        # If linear interpolation fails, use nearest neighbor
        if np.isnan(z_interp[0]):
            z_interp = griddata(
                points=all_coords[:, :2],
                values=all_coords[:, 2],
                xi=np.array([[x_rd, y_rd]]),
                method='nearest'
            )
        
        z_value = float(z_interp[0])
        
        # Transform back to WGS84
        lon, lat = transformer_to_wgs.transform(x_rd, y_rd)
        coords_3d.append([lon, lat, z_value])
    
    # Create 3D feature
    feature_3d = {
        'type': 'Feature',
        'geometry': {
            'type': 'Polygon',
            'coordinates': [coords_3d]
        },
        'properties': {
            'polygon_id': poly_idx,
            'area_2d_m2': round(poly_rd.area, 2),
            'type': '3D ruimtebeslag'
        }
    }
    features_3d.append(feature_3d)

# Create GeoJSON FeatureCollection
geojson_3d = {
    'type': 'FeatureCollection',
    'crs': {
        'type': 'name',
        'properties': {'name': 'EPSG:4326'}
    },
    'features': features_3d
}

# Save to file
  # 3D output path removed
with open(output_path, 'w') as f:
    json.dump(geojson_3d, f, indent=2)

print(f"\n3D ruimtebeslag GeoJSON saved to: {output_path}")
print(f"  Number of polygons: {len(features_3d)}")
print(f"  Total 2D area: {sum(p['properties']['area_2d_m2'] for p in features_3d):.2f} m²")

# Show sample coordinates
if features_3d:
    sample_coords = features_3d[0]['geometry']['coordinates'][0][:3]
    print(f"\n  Sample coordinates (lon, lat, elevation):")
    for coord in sample_coords:
        print(f"    [{coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.2f}]")
