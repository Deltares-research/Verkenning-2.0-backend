"""
Generate 3D ruimtebeslag GeoJSON with Z-coordinates
"""
import sys
import importlib
from http.client import HTTPException
from urllib import request

from scipy.spatial import Delaunay

from app.dike_components.ground_model import GroundModel
from main import VolumeCalculationResult, DesignCalculationResult

sys.path.insert(0, '..')

import json
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from pyproj import Transformer
from scipy.interpolate import griddata
# Test GeoJSON input


payload1 = {
    "geojson":{
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
              5.583301858567071,
              51.89199846805985,
              7.9
            ],
            [
              5.584044761674879,
              51.89263258718884,
              7.9
            ],
            [
              5.583863625831217,
              51.89271382567278,
              11.6
            ],
            [
              5.583120723932436,
              51.892079705404406,
              11.6
            ],
            [
              5.583301858567071,
              51.89199846805985,
              7.9
            ]
          ]
        ]
      },
      "properties": {
        "name": "-36.8m_-21.4m"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              5.583120723932436,
              51.892079705404406,
              11.6
            ],
            [
              5.583863625831217,
              51.89271382567278,
              11.6
            ],
            [
              5.58370954222817,
              51.89278293091615,
              12.5
            ],
            [
              5.58296664135787,
              51.89214880967859,
              12.5
            ],
            [
              5.583120723932436,
              51.892079705404406,
              11.6
            ]
          ]
        ]
      },
      "properties": {
        "name": "-21.4m_-8.3m"
      }
    }
  ]
}
}

payload2 = {   # test_data/test_berm__onderwerp_3d
    "geojson": {
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
              5.5880257283351,
              51.89317402584419,
              8.5
            ],
            [
              5.5894349916198305,
              51.893210674409914,
              8.5
            ],
            [
              5.590070111617253,
              51.893406527779895,
              8.5
            ],
            [
              5.590225087768538,
              51.893546975044146,
              8.5
            ],
            [
              5.59016749740108,
              51.893571301269766,
              10.3
            ],
            [
              5.59002275777441,
              51.89344013074613,
              10.3
            ],
            [
              5.589417200422898,
              51.893253393625294,
              10.3
            ],
            [
              5.5880227993282805,
              51.893217131470266,
              10.3
            ],
            [
              5.5880257283351,
              51.89317402584419,
              8.5
            ]
          ]
        ]
      },
      "properties": {
        "name": "-27.2m_-22.4m"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              5.5880227993282805,
              51.893217131470266,
              10.3
            ],
            [
              5.589417200422898,
              51.893253393625294,
              10.3
            ],
            [
              5.59002275777441,
              51.89344013074613,
              10.3
            ],
            [
              5.59016749740108,
              51.893571301269766,
              10.3
            ],
            [
              5.589998325336215,
              51.893642759394,
              11
            ],
            [
              5.589883655451839,
              51.89353883934842,
              11
            ],
            [
              5.589364938586728,
              51.893378881303704,
              11
            ],
            [
              5.588014195338595,
              51.893343754245045,
              11
            ],
            [
              5.5880227993282805,
              51.893217131470266,
              10.3
            ]
          ]
        ]
      },
      "properties": {
        "name": "-22.4m_-8.3m"
      }
    }
  ]
}
}

payload3 = {
    "geojson" : {
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
              5.620224118292769,
              51.89131086133689,
              8.3
            ],
            [
              5.62070448018626,
              51.890779792025604,
              8.3
            ],
            [
              5.621303933304355,
              51.89034854669813,
              8.3
            ],
            [
              5.62181051133016,
              51.8901614399288,
              8.3
            ],
            [
              5.621825402630816,
              51.89017687550451,
              9.2
            ],
            [
              5.621322880000625,
              51.890362484433425,
              9.2
            ],
            [
              5.620728408559097,
              51.890790146092286,
              9.2
            ],
            [
              5.620249471223593,
              51.89131964077616,
              9.2
            ],
            [
              5.620224118292769,
              51.89131086133689,
              8.3
            ]
          ]
        ]
      },
      "properties": {
        "name": "-26.9m_-24.9m"
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              5.620249471223593,
              51.89131964077616,
              9.2
            ],
            [
              5.620728408559097,
              51.890790146092286,
              9.2
            ],
            [
              5.621322880000625,
              51.890362484433425,
              9.2
            ],
            [
              5.621825402630816,
              51.89017687550451,
              9.2
            ],
            [
              5.6219169843547085,
              51.89027180425284,
              10
            ],
            [
              5.621439402440937,
              51.89044820143741,
              10
            ],
            [
              5.620875568294213,
              51.89085382349476,
              10
            ],
            [
              5.620405391965653,
              51.891373634207085,
              10
            ],
            [
              5.620249471223593,
              51.89131964077616,
              9.2
            ]
          ]
        ]
      },
      "properties": {
        "name": "-24.9m_-12.6m"
      }
    }
  ]
}
}

import time

start_time = time.time()
TIN_MAX_EDGE_FACTOR = 8.0


def filter_triangles_by_max_edge_length(points_xy: np.ndarray, triangles: np.ndarray, max_edge_length: float) -> np.ndarray:
  if triangles.size == 0 or max_edge_length <= 0:
    return triangles

  tri_pts = points_xy[triangles]
  edge01 = np.linalg.norm(tri_pts[:, 0, :] - tri_pts[:, 1, :], axis=1)
  edge12 = np.linalg.norm(tri_pts[:, 1, :] - tri_pts[:, 2, :], axis=1)
  edge20 = np.linalg.norm(tri_pts[:, 2, :] - tri_pts[:, 0, :], axis=1)
  max_edge_per_triangle = np.maximum(np.maximum(edge01, edge12), edge20)
  keep_mask = max_edge_per_triangle <= max_edge_length
  return triangles[keep_mask]


def plot_payload_polygons_3d(payload_data: dict, title: str = "Payload polygons (3D)", max_edge_length: float = 0.0):
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')

  min_x, max_x = np.inf, -np.inf
  min_y, max_y = np.inf, -np.inf
  min_z, max_z = np.inf, -np.inf

  for feature in payload_data['geojson']['features']:
    geometry = feature.get('geometry', {})
    if geometry.get('type') != 'Polygon':
      continue

    rings = geometry.get('coordinates', [])
    if not rings:
      continue

    exterior_ring = rings[0]
    if len(exterior_ring) < 4:
      continue

    if len(exterior_ring[0]) < 3:
      raise ValueError("All payload polygon coordinates must include Z values")

    if exterior_ring[0] == exterior_ring[-1]:
      exterior_ring = exterior_ring[:-1]

    if len(exterior_ring) < 3:
      continue

    xs_lon, ys_lat, zs = [], [], []
    for lon, lat, z in exterior_ring:
      xs_lon.append(lon)
      ys_lat.append(lat)
      zs.append(z)

    triangulation = mtri.Triangulation(np.array(xs_lon), np.array(ys_lat))
    if triangulation.triangles.size > 0:
      tri_points_xy = np.column_stack([xs_lon, ys_lat])
      filtered_triangles = filter_triangles_by_max_edge_length(tri_points_xy, triangulation.triangles, max_edge_length)
      polygon_path = Path(np.column_stack([xs_lon, ys_lat]))
      triangles = filtered_triangles if filtered_triangles.size > 0 else triangulation.triangles
      centroids = np.array([
        [
          (xs_lon[t[0]] + xs_lon[t[1]] + xs_lon[t[2]]) / 3.0,
          (ys_lat[t[0]] + ys_lat[t[1]] + ys_lat[t[2]]) / 3.0
        ]
        for t in triangles
      ])
      outside_mask = ~polygon_path.contains_points(centroids)
      inside_triangles = triangles[~outside_mask]
      if inside_triangles.size == 0:
        continue
      triangulation = mtri.Triangulation(np.array(xs_lon), np.array(ys_lat), triangles=inside_triangles)

      ax.plot_trisurf(
        triangulation,
        np.array(zs),
        color='tab:blue',
        alpha=0.45,
        linewidth=0.0,
        antialiased=True,
        shade=True
      )

    xs_closed = xs_lon + [xs_lon[0]]
    ys_closed = ys_lat + [ys_lat[0]]
    zs_closed = zs + [zs[0]]
    ax.plot(xs_closed, ys_closed, zs_closed, color='black', linewidth=1)

    min_x, max_x = min(min_x, min(xs_lon)), max(max_x, max(xs_lon))
    min_y, max_y = min(min_y, min(ys_lat)), max(max_y, max(ys_lat))
    min_z, max_z = min(min_z, min(zs)), max(max_z, max(zs))

  if np.isinf(min_x):
    raise ValueError("No valid Polygon geometries found in payload")

  mean_lat_rad = np.deg2rad((min_y + max_y) / 2.0)
  meters_per_degree_lat = 111320.0
  meters_per_degree_lon = meters_per_degree_lat * np.cos(mean_lat_rad)
  range_x = max((max_x - min_x) * meters_per_degree_lon, 1e-6)
  range_y = max((max_y - min_y) * meters_per_degree_lat, 1e-6)
  range_z = max(max_z - min_z, 1e-6)

  ax.set_xlim(min_x, max_x)
  ax.set_ylim(min_y, max_y)
  ax.set_zlim(min_z, max_z)
  ax.set_box_aspect((range_x, range_y, range_z))

  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.set_zlabel('Elevation (m)')
  ax.set_title(title)
  plt.tight_layout()
  plt.show()


def plot_payload_polygons_3d_plotly(payload_data: dict, title: str = "Payload polygons (3D) - Plotly", max_edge_length: float = 0.0):
  try:
    go = importlib.import_module("plotly.graph_objects")
  except ImportError as exc:
    raise ImportError("Plotly is not installed. Install with: pip install plotly") from exc

  min_x, max_x = np.inf, -np.inf
  min_y, max_y = np.inf, -np.inf
  min_z, max_z = np.inf, -np.inf
  traces = []

  for feature in payload_data['geojson']['features']:
    geometry = feature.get('geometry', {})
    if geometry.get('type') != 'Polygon':
      continue

    rings = geometry.get('coordinates', [])
    if not rings:
      continue

    exterior_ring = rings[0]
    if len(exterior_ring) < 4:
      continue

    if len(exterior_ring[0]) < 3:
      raise ValueError("All payload polygon coordinates must include Z values")

    if exterior_ring[0] == exterior_ring[-1]:
      exterior_ring = exterior_ring[:-1]

    if len(exterior_ring) < 3:
      continue

    xs_lon, ys_lat, zs = [], [], []
    for lon, lat, z in exterior_ring:
      xs_lon.append(lon)
      ys_lat.append(lat)
      zs.append(z)

    triangulation = mtri.Triangulation(np.array(xs_lon), np.array(ys_lat))
    if triangulation.triangles.size > 0:
      tri_points_xy = np.column_stack([xs_lon, ys_lat])
      filtered_triangles = filter_triangles_by_max_edge_length(tri_points_xy, triangulation.triangles, max_edge_length)
      polygon_path = Path(np.column_stack([xs_lon, ys_lat]))
      triangles = filtered_triangles if filtered_triangles.size > 0 else triangulation.triangles
      centroids = np.array([
        [
          (xs_lon[t[0]] + xs_lon[t[1]] + xs_lon[t[2]]) / 3.0,
          (ys_lat[t[0]] + ys_lat[t[1]] + ys_lat[t[2]]) / 3.0
        ]
        for t in triangles
      ])
      outside_mask = ~polygon_path.contains_points(centroids)
      valid_triangles = triangles[~outside_mask]

      if len(valid_triangles) > 0:
        traces.append(
          go.Mesh3d(
            x=xs_lon,
            y=ys_lat,
            z=zs,
            i=valid_triangles[:, 0],
            j=valid_triangles[:, 1],
            k=valid_triangles[:, 2],
            color='royalblue',
            opacity=0.45,
            name=feature.get('properties', {}).get('name', 'polygon'),
            showscale=False,
            flatshading=False
          )
        )

    xs_closed = xs_lon + [xs_lon[0]]
    ys_closed = ys_lat + [ys_lat[0]]
    zs_closed = zs + [zs[0]]
    traces.append(
      go.Scatter3d(
        x=xs_closed,
        y=ys_closed,
        z=zs_closed,
        mode='lines',
        line=dict(color='black', width=4),
        name=f"edge-{feature.get('properties', {}).get('name', '')}",
        showlegend=False
      )
    )

    min_x, max_x = min(min_x, min(xs_lon)), max(max_x, max(xs_lon))
    min_y, max_y = min(min_y, min(ys_lat)), max(max_y, max(ys_lat))
    min_z, max_z = min(min_z, min(zs)), max(max_z, max(zs))

  if np.isinf(min_x):
    raise ValueError("No valid Polygon geometries found in payload")

  mean_lat_rad = np.deg2rad((min_y + max_y) / 2.0)
  meters_per_degree_lat = 111320.0
  meters_per_degree_lon = meters_per_degree_lat * np.cos(mean_lat_rad)
  range_x = max((max_x - min_x) * meters_per_degree_lon, 1e-6)
  range_y = max((max_y - min_y) * meters_per_degree_lat, 1e-6)
  range_z = max(max_z - min_z, 1e-6)

  fig = go.Figure(data=traces)
  fig.update_layout(
    title=title,
    scene=dict(
      xaxis_title='Longitude',
      yaxis_title='Latitude',
      zaxis_title='Elevation (m)',
      xaxis=dict(range=[min_x, max_x]),
      yaxis=dict(range=[min_y, max_y]),
      zaxis=dict(range=[min_z, max_z]),
      aspectmode='manual',
      aspectratio=dict(x=range_x, y=range_y, z=range_z)
    ),
    margin=dict(l=0, r=0, t=40, b=0)
  )
  fig.show()



# Convert GeoJSON to GeoDataFrame
features = []
active_payload = payload2
for feature in active_payload['geojson']['features']:
    geom = shape(feature['geometry'])
    features.append({
        'geometry': geom,
        **feature['properties']
    })

gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")

# Check if geometries are 3D
if not gdf.geometry.iloc[0].has_z:
    raise HTTPException(
        status_code=400,
        detail="Geometry must be 3D (include Z coordinates for elevation)"
    )

# Initialize GroundModel with the GeoDataFrame
ground_model = GroundModel(gdf, excavation_mode="envelope")

# Calculate volume using Matthias's method
volume_start = time.time()
result = ground_model.calculate_volume()

print(f"DEBUG: Result type: {type(result)}")
print(f"DEBUG: Result value: {result}")

# If result is None, extract from ground_model attributes
if isinstance(result, dict):
    print(f"DEBUG: Result is dict: {result}")
    fill_vol = result.get('fill_volume', 0.0)
    cut_vol = result.get('cut_volume', 0.0)
    total_vol = result.get('net_volume', 0.0)
    area = result.get('area', 0.0)
    grid_pts = result.get('grid_points', None)
else:
    raise ValueError(f"Unexpected result type: {type(result)}")

volume_time = time.time() - volume_start

# Calculate 2D ruimtebeslag
ruimtebeslag_result = ground_model.calculate_ruimtebeslag_2d()

total_calculation_time = time.time() - start_time

# Build volume calculation result
volume_calc = VolumeCalculationResult(
    net_volume=round(total_vol, 2),
    excavation_volume=round(cut_vol, 2),
    fill_volume=round(fill_vol, 2),
    area=round(area, 2),
    calculation_time=round(volume_time, 3),
    grid_points=grid_pts
)

# Build ruimtebeslag GeoJSON with metadata
ruimtebeslag_points = ruimtebeslag_result['points_above_ground']
# ruimtebeslag_2d_points can now be a list as per the updated Pydantic model
a = DesignCalculationResult(
    volume=volume_calc,
    calculation_time=round(total_calculation_time, 3),
    ruimtebeslag_2d_points=ruimtebeslag_points)

print("2D ruimtebeslag)", ruimtebeslag_result["total_area_m2"])


# dict = ground_model.calculate_all_dike_volumes()
# print(dict)


method1 = ground_model.calculate_3d_surface_TIN(height_source='design')
method2 = ground_model.calculate_total_3d_surface_area()['total_3d_area_m2']

print("TIN design:", method1, " =3D area of design")
print("Newells design:", method2, " =3D area of design")


valid = ~np.isnan(ground_model.elev_global)
points_xy = ground_model.grid_pts_global[valid]
points_z = ground_model.elev_global[valid]

points_3d = np.column_stack((points_xy, points_z))  # (N,3)

points_xy = points_3d[:, :2]
triangles = np.empty((0, 3), dtype=int)
largest_triangle_indices = np.array([], dtype=int)
triangle_areas = np.array([], dtype=float)

if points_xy.shape[0] >= 3:
  try:
    tri = Delaunay(points_xy)
    triangles = tri.simplices  # indices of triangle vertices
    max_edge_length = TIN_MAX_EDGE_FACTOR * ground_model.grid_size
    triangles = filter_triangles_by_max_edge_length(points_xy, triangles, max_edge_length)

    if triangles.size > 0:
      p0 = points_xy[triangles[:, 0]]
      p1 = points_xy[triangles[:, 1]]
      p2 = points_xy[triangles[:, 2]]
      double_area = np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
        (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
      )
      triangle_areas = 0.5 * double_area

      top_k = min(3, triangle_areas.shape[0])
      largest_triangle_indices = np.argsort(triangle_areas)[-top_k:][::-1]
  except Exception as exc:
    print(f"Could not compute Delaunay triangles for top-area plotting: {exc}")




def infer_xy_crs(points_xy_values: np.ndarray) -> str:
  if points_xy_values.size == 0:
    return "EPSG:4326"

  x_vals = points_xy_values[:, 0]
  y_vals = points_xy_values[:, 1]
  looks_like_lon_lat = (
    np.all((x_vals >= -180.0) & (x_vals <= 180.0)) and
    np.all((y_vals >= -90.0) & (y_vals <= 90.0))
  )
  return "EPSG:4326" if looks_like_lon_lat else "EPSG:28992"

if points_3d.size == 0:
  print("No 3D points available to plot.")
else:
  points_crs = infer_xy_crs(points_xy)
  payload_crs = active_payload.get('geojson', {}).get('crs', {}).get('properties', {}).get('name', 'EPSG:4326')
  if not isinstance(payload_crs, str) or not payload_crs:
    payload_crs = 'EPSG:4326'

  payload_to_points_transformer = None
  if payload_crs != points_crs:
    payload_to_points_transformer = Transformer.from_crs(payload_crs, points_crs, always_xy=True)

  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')
  scatter = ax.scatter(
    points_3d[:, 0],
    points_3d[:, 1],
    points_3d[:, 2],
    c=points_3d[:, 2],
    cmap='viridis',
    s=8,
    alpha=0.9
  )

  # Overlay the 3 largest triangles from the point cloud triangulation
  for rank, tri_idx in enumerate(largest_triangle_indices, start=1):
    vertex_idx = triangles[tri_idx]
    tri_pts = points_3d[vertex_idx]
    tri_closed = np.vstack([tri_pts, tri_pts[0]])

    ax.plot(
      tri_closed[:, 0],
      tri_closed[:, 1],
      tri_closed[:, 2],
      color='black',
      linewidth=4.0
    )
    ax.plot_trisurf(
      tri_pts[:, 0],
      tri_pts[:, 1],
      tri_pts[:, 2],
      triangles=[[0, 1, 2]],
      color='gold',
      alpha=0.2,
      shade=False
    )

    centroid = tri_pts.mean(axis=0)
    ax.text(
      centroid[0],
      centroid[1],
      centroid[2],
      f"T{rank}",
      color='black'
    )

  if largest_triangle_indices.size > 0:
    top_areas_str = ", ".join([f"{triangle_areas[i]:.2f}" for i in largest_triangle_indices])
    print(f"Top triangle areas: {top_areas_str}")

  # Overlay payload polygons on the same 3D axes
  for feature in active_payload['geojson']['features']:
    geometry = feature.get('geometry', {})
    if geometry.get('type') != 'Polygon':
      continue

    rings = geometry.get('coordinates', [])
    if not rings:
      continue

    exterior_ring = rings[0]
    if len(exterior_ring) < 4:
      continue

    if exterior_ring[0] == exterior_ring[-1]:
      exterior_ring = exterior_ring[:-1]

    if len(exterior_ring) < 3:
      continue

    if payload_to_points_transformer is None:
      xs_poly = [p[0] for p in exterior_ring]
      ys_poly = [p[1] for p in exterior_ring]
    else:
      transformed_xy = [payload_to_points_transformer.transform(p[0], p[1]) for p in exterior_ring]
      xs_poly = [p[0] for p in transformed_xy]
      ys_poly = [p[1] for p in transformed_xy]
    zs_poly = [p[2] for p in exterior_ring]

    triangulation = mtri.Triangulation(np.array(xs_poly), np.array(ys_poly))
    if triangulation.triangles.size > 0:
      tri_points_xy = np.column_stack([xs_poly, ys_poly])
      max_edge_length = TIN_MAX_EDGE_FACTOR * ground_model.grid_size
      filtered_triangles = filter_triangles_by_max_edge_length(tri_points_xy, triangulation.triangles, max_edge_length)
      polygon_path = Path(np.column_stack([xs_poly, ys_poly]))
      payload_triangles = filtered_triangles if filtered_triangles.size > 0 else triangulation.triangles
      centroids = np.array([
        [
          (xs_poly[t[0]] + xs_poly[t[1]] + xs_poly[t[2]]) / 3.0,
          (ys_poly[t[0]] + ys_poly[t[1]] + ys_poly[t[2]]) / 3.0
        ]
        for t in payload_triangles
      ])
      outside_mask = ~polygon_path.contains_points(centroids)
      inside_triangles = payload_triangles[~outside_mask]
      if inside_triangles.size == 0:
        continue
      triangulation = mtri.Triangulation(np.array(xs_poly), np.array(ys_poly), triangles=inside_triangles)

      ax.plot_trisurf(
        triangulation,
        np.array(zs_poly),
        color='tomato',
        alpha=0.35,
        linewidth=0.0,
        antialiased=True,
        shade=True
      )

    ax.plot(
      xs_poly + [xs_poly[0]],
      ys_poly + [ys_poly[0]],
      zs_poly + [zs_poly[0]],
      color='darkred',
      linewidth=1.5
    )

  if points_crs == 'EPSG:4326':
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
  else:
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
  ax.set_zlabel('Z')
  ax.set_title('3D scatter of points_3d + payload polygons')
  fig.colorbar(scatter, ax=ax, shrink=0.65, label='Z')
  plt.tight_layout()
  plt.show()

# # Plot payload polygons in 3D
# plot_backend = "matplotlib"  # options: "matplotlib", "plotly", "both"
# if plot_backend == "matplotlib":
#   plot_payload_polygons_3d(active_payload, title="Input payload polygons (3D)")
# elif plot_backend == "plotly":
#   plot_payload_polygons_3d_plotly(active_payload, title="Input payload polygons (3D) - Plotly")
# elif plot_backend == "both":
#   plot_payload_polygons_3d(active_payload, title="Input payload polygons (3D)")
#   plot_payload_polygons_3d_plotly(active_payload, title="Input payload polygons (3D) - Plotly")
