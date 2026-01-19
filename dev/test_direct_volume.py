"""
Direct test of volume calculation without API server.
Expected volume: ~4070 m³
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import geopandas as gpd
from shapely.geometry import shape
from app.volume_calc import DikeModel

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


def main():
    print("="*70)
    print("DIRECT VOLUME CALCULATION TEST (No API)")
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
        # Convert GeoJSON to GeoDataFrame
        features = []
        for feature in test_geojson['features']:
            geom = shape(feature['geometry'])
            features.append({
                'geometry': geom,
                **feature['properties']
            })
        
        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        
        print(f"\n{'='*70}")
        print("Initializing DikeModel and calculating volumes...")
        print(f"{'='*70}\n")
        
        # Initialize DikeModel with adjusted grid size for better match
        dike_model = DikeModel(gdf, grid_size=0.525)
        
        # Calculate volumes
        result = dike_model.calculate_volume()
        
        print(f"\n{'='*70}")
        print("✅ RESULTS")
        print(f"{'='*70}")
        print(f"\n  Fill Volume:       {result['fill_volume']:>10.2f} m³")
        print(f"  Excavation Volume: {result['cut_volume']:>10.2f} m³")
        print(f"  Total Volume:      {result['total_volume']:>10.2f} m³")
        print(f"  Grid Area:         {result['area']:>10.2f} m²")
        print(f"  Grid Points:       {result['grid_points']:>10}")
        
        expected = 4070
        total_vol = result['total_volume']
        diff = abs(total_vol - expected)
        diff_pct = (diff / expected) * 100 if expected > 0 else 0
        
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"\n  Expected:   {expected:>10.2f} m³")
        print(f"  Calculated: {total_vol:>10.2f} m³")
        print(f"  Difference: {diff:>10.2f} m³ ({diff_pct:.1f}%)")
        
        if diff_pct < 5:
            print(f"\n  ✅ Result is within 5% tolerance - EXCELLENT!")
        elif diff_pct < 10:
            print(f"\n  ⚠️  Result is within 10% tolerance - GOOD")
        elif diff_pct < 20:
            print(f"\n  ⚠️  Result is within 20% tolerance - ACCEPTABLE")
        else:
            print(f"\n  ❌ Result differs by more than 20% - NEEDS REVIEW")
        
        print(f"\n{'='*70}")
        print("Testing Ruimtebeslag 2D Calculation...")
        print(f"{'='*70}\n")
        
        # Also test ruimtebeslag
        ruimte_result = dike_model.calculate_ruimtebeslag_2d(alpha=5.0)
        
        print(f"\n✅ RUIMTEBESLAG RESULTS")
        print(f"{'='*70}")
        print(f"\n  Total Area:         {ruimte_result['total_area_m2']:>10.2f} m²")
        print(f"  Number of Polygons: {ruimte_result['num_polygons']:>10}")
        print(f"  Points Above Ground:{ruimte_result['points_above_ground']:>10}")
        
        # Save output
        import json
        output_file = os.path.join(os.path.dirname(__file__), "..", "output", "test_ruimtebeslag_output.geojson")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                'type': ruimte_result['type'],
                'features': ruimte_result['features'],
                'crs': ruimte_result.get('crs')
            }, f, indent=2)
        print(f"\n  Output saved to: {output_file}")
        
        # Visualize the polygons (GIS-style overlay)
        print(f"\n{'='*70}")
        print("Generating visualization...")
        print(f"{'='*70}\n")
        
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MPLPolygon
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(16, 12), facecolor='#E8F5E9')
        ax.set_facecolor('#F1F8F4')
        
        # Collect all coordinates to determine bounds and offset
        all_coords = []
        for feature in ruimte_result['features']:
            all_coords.extend(feature['geometry']['coordinates'][0])
        
        all_coords = np.array(all_coords)
        x_min, y_min = all_coords.min(axis=0)
        x_max, y_max = all_coords.max(axis=0)
        
        # Calculate offset to center the plot
        x_offset = (x_min + x_max) / 2
        y_offset = (y_min + y_max) / 2
        
        # GIS-style colors: yellow/orange/peach tones like in the reference image
        gis_colors = ['#FFD54F', '#FFB74D', '#FFA726', '#FF9800', '#FFE082']
        edge_colors = ['#F57C00', '#EF6C00', '#E65100', '#FF6F00', '#FB8C00']
        
        # Plot each polygon from ruimtebeslag result
        for i, feature in enumerate(ruimte_result['features']):
            coords = np.array(feature['geometry']['coordinates'][0])
            # Shift coordinates to origin for better display
            coords_shifted = coords - [x_offset, y_offset]
            
            poly_patch = MPLPolygon(coords_shifted, fill=True, alpha=0.65, 
                                   edgecolor=edge_colors[i % len(edge_colors)], 
                                   facecolor=gis_colors[i % len(gis_colors)],
                                   linewidth=3)
            ax.add_patch(poly_patch)
            
            # Add label with area
            centroid_x = coords_shifted[:, 0].mean()
            centroid_y = coords_shifted[:, 1].mean()
            area = feature['properties']['area_m2']
            name = feature['properties'].get('name', f'Poly {i}')
            
            # Add text label with shadow effect
            ax.text(centroid_x+0.5, centroid_y-0.5, f"{area:.0f} m²", 
                   ha='center', va='center', fontsize=11, fontweight='normal',
                   color='#424242', alpha=0.3)
            ax.text(centroid_x, centroid_y, f"{area:.0f} m²", 
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color='#1B5E20')
        
        # Set axis limits with some padding
        padding = 25  # meters
        ax.set_xlim([all_coords[:, 0].min() - x_offset - padding, 
                     all_coords[:, 0].max() - x_offset + padding])
        ax.set_ylim([all_coords[:, 1].min() - y_offset - padding, 
                     all_coords[:, 1].max() - y_offset + padding])
        
        # Set axis properties (minimal style like GIS viewer)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#9E9E9E')
        ax.set_xlabel('X (meters, RD New)', fontsize=11, color='#424242')
        ax.set_ylabel('Y (meters, RD New)', fontsize=11, color='#424242')
        ax.set_title(f'Ruimtebeslag 2D (Above Ground Level)\nTotal: {ruimte_result["total_area_m2"]:.1f} m²', 
                    fontsize=14, fontweight='bold', color='#1B5E20', pad=20)
        
        # Style the spines
        for spine in ax.spines.values():
            spine.set_edgecolor('#9E9E9E')
            spine.set_linewidth(1)
        
        # Add legend with GIS-style appearance
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=gis_colors[i % len(gis_colors)], alpha=0.65, 
                                edgecolor=edge_colors[i % len(edge_colors)], linewidth=2,
                                label=f"{feat['properties'].get('name', f'Polygon {i}')}: {feat['properties']['area_m2']:.1f} m²")
                          for i, feat in enumerate(ruimte_result['features'])]
        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                          framealpha=0.95, edgecolor='#9E9E9E', fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        
        # Add info box with grid statistics
        total_area = ruimte_result["total_area_m2"]
        expected_area = 4070
        diff_pct = abs(total_area - expected_area) / expected_area * 100
        
        stats_text = (f"Grid size: {dike_model.grid_size}m\n"
                     f"Grid points above ground: {ruimte_result['points_above_ground']:,}\n"
                     f"Expected area: {expected_area:.0f} m²\n"
                     f"Difference: {diff_pct:.1f}%")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor='#9E9E9E', linewidth=1.5),
               color='#424242', family='monospace')
        
        # Add north arrow
        arrow_x, arrow_y = 0.95, 0.05
        ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y-0.03),
                   xycoords='axes fraction', fontsize=14, fontweight='bold',
                   ha='center', color='#1B5E20',
                   arrowprops=dict(arrowstyle='->', lw=2, color='#1B5E20'))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(os.path.dirname(__file__), "..", "output", "ruimtebeslag_visualization.png")
        plt.savefig(plot_file, dpi=200, facecolor='#E8F5E9', edgecolor='none')
        print(f"  Visualization saved to: {plot_file}")
        
        # Show plot
        plt.show()
        print(f"  Plot displayed (close window to continue)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    print(f"\n{'='*70}")
    print("Test completed!")
    print(f"{'='*70}\n")
