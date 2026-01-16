"""Check the actual areas of input polygons"""
from shapely.geometry import Polygon
import numpy as np

# Input polygons
polygons = [
    {
        "name": "-30.4m_-19.5m",
        "coords": [
            [5.605224768153091, 51.895764079821106, 7.5],
            [5.607046480372475, 51.89510601605683, 7.5],
            [5.60786772690824, 51.894623004573546, 7.5],
            [5.607976831307955, 51.89469401159619, 11.2],
            [5.607142234599198, 51.89518487547184, 11.2],
            [5.605304613562746, 51.89584868666671, 11.2],
        ],
        "stated_area_3d": 1596.5098616451694
    },
    {
        "name": "-19.5m_11.9m",
        "coords": [
            [5.605304613562746, 51.89584868666671, 11.2],
            [5.607142234599198, 51.89518487547184, 11.2],
            [5.607976831307955, 51.89469401159619, 11.2],
            [5.608291133981822, 51.89489856336535, 10.8],
            [5.607418078928262, 51.89541204802368, 10.8],
            [5.605534628625057, 51.89609241616928, 10.8],
        ],
        "stated_area_3d": 4009.142985984193
    },
    {
        "name": "11.9m_17.4m",
        "coords": [
            [5.605534628625057, 51.89609241616928, 10.8],
            [5.607418078928262, 51.89541204802368, 10.8],
            [5.608291133981822, 51.89489856336535, 10.8],
            [5.608346187292249, 51.89493439240966, 7.5],
            [5.607466395897025, 51.89545183932671, 7.5],
            [5.605574918144323, 51.89613510759537, 7.5],
        ],
        "stated_area_3d": 235.94413436814
    }
]

# Convert WGS84 to RD (EPSG:28992)
from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)

print("=== INPUT POLYGON AREAS ===\n")
total_2d = 0
total_3d_calculated = 0

for poly_data in polygons:
    # Transform to RD
    coords_rd = []
    for lon, lat, z in poly_data["coords"]:
        x, y = transformer.transform(lon, lat)
        coords_rd.append((x, y, z))
    
    # Create 2D and 3D polygons
    poly_2d = Polygon([(x, y) for x, y, z in coords_rd])
    poly_3d = Polygon(coords_rd)
    
    # Calculate 3D area (triangulation)
    coords = np.array(coords_rd)
    total_3d = 0.0
    p0 = coords[0]
    for i in range(1, len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        v1 = p1 - p0
        v2 = p2 - p0
        cross = np.cross(v1, v2)
        total_3d += 0.5 * np.linalg.norm(cross)
    
    total_2d += poly_2d.area
    total_3d_calculated += total_3d
    
    print(f"{poly_data['name']}:")
    print(f"  2D footprint: {poly_2d.area:.2f} m²")
    print(f"  3D calculated: {total_3d:.2f} m²")
    print(f"  3D stated: {poly_data['stated_area_3d']:.2f} m²")
    print(f"  Difference: {abs(total_3d - poly_data['stated_area_3d']):.2f} m²")
    print()

print(f"TOTALS:")
print(f"  Total 2D: {total_2d:.2f} m²")
print(f"  Total 3D calculated: {total_3d_calculated:.2f} m²")
print(f"  Total 3D stated in properties: 5841.59 m²")
print(f"\nCalculated from backend:")
print(f"  2D footprint: 10,934.59 m²")
print(f"  3D surface: 19,357.23 m²")
