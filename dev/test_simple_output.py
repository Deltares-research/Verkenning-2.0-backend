"""
Simple test to show ruimtebeslag GeoJSON output directly
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import geopandas as gpd
from shapely.geometry import shape
from app.volume_calc import DikeModel
import json

# Input data from user
test_geojson = {
  "type": "FeatureCollection",
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
      "properties": {"name": "-26.3m_-16.4m", "area_3d": 1213.309092251679}
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
      "properties": {"name": "-16.4m_8.2m", "area_3d": 2189.3953221486363}
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
      "properties": {"name": "8.2m_15.9m", "area_3d": 783.6177734971092}
    }
  ]
}


def main():
    print("="*70)
    print("RUIMTEBESLAG OUTPUT TEST")
    print("="*70)
    
    # Convert GeoJSON to GeoDataFrame
    features = []
    for feature in test_geojson['features']:
        geom = shape(feature['geometry'])
        features.append({'geometry': geom, **feature['properties']})
    
    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    
    # Initialize DikeModel
    dike_model = DikeModel(gdf, grid_size=0.525)
    
    # Calculate ruimtebeslag
    print("\nCalculating ruimtebeslag with alpha=2.0...")
    result = dike_model.calculate_ruimtebeslag_2d(alpha=2.0)
    
    print(f"\n{'='*70}")
    print("RESULT SUMMARY")
    print(f"{'='*70}")
    print(f"Total Area: {result['total_area_m2']:.2f} m²")
    print(f"Number of Polygons: {result['num_polygons']}")
    print(f"Points Above Ground: {result['points_above_ground']}")
    
    # Save to file
    output_file = os.path.join(os.path.dirname(__file__), "..", "output", "ruimtebeslag_output.geojson")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Output saved to: {output_file}")
    
    # Print first polygon as sample
    if result['features']:
        print(f"\n{'='*70}")
        print("SAMPLE: First Polygon")
        print(f"{'='*70}")
        first_feature = result['features'][0]
        print(f"Area: {first_feature['properties']['area_m2']:.2f} m²")
        print(f"Number of coordinates: {len(first_feature['geometry']['coordinates'][0])}")
        print(f"\nFirst 5 coordinates (WGS84):")
        for i, coord in enumerate(first_feature['geometry']['coordinates'][0][:5]):
            print(f"  {i+1}. [{coord[0]:.12f}, {coord[1]:.14f}]")
    
    print(f"\n{'='*70}")
    print("Open the output file to see the full GeoJSON")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
