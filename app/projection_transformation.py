# Convert WGS84 to RD (EPSG:28992)
from pyproj import Transformer

def transform_to_rd(longitude, latitude):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    return transformer.transform(longitude, latitude)

