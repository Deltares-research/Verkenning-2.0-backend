from shapely.geometry.polygon import Polygon
from pyproj import Transformer



transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
transformer_rd_to_wm = Transformer.from_crs("EPSG:28992", "EPSG:3857", always_xy=True)

def reproject_polygon_with_z(poly):
    """
    Reproject a Polygon Z from WGS84 to RD while keeping the Z coordinate.
    Returns a Polygon with 3D coordinates (x_RD, y_RD, z).
    """
    exterior_coords_3d = []
    for x, y, *z in poly.exterior.coords:
        z_val = z[0] if z else 0.0
        x_rd, y_rd = transformer.transform(x, y)
        exterior_coords_3d.append((x_rd, y_rd, z_val))
    # Ensure the polygon is closed
    if exterior_coords_3d[0] != exterior_coords_3d[-1]:
        exterior_coords_3d.append(exterior_coords_3d[0])
    return Polygon(exterior_coords_3d)