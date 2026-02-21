import requests
import numpy as np
from shapely.geometry import LineString
from scipy.ndimage import map_coordinates

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False


"""
AHN client for extracting elevation profiles along LineStrings
using an Esri ImageServer (exportImage endpoint).

Features:
- Raster caching to avoid repeated downloads
- Small bbox computed from a LineString with configurable buffer
- Uses rasterio for fast GeoTIFF decoding
- Uses scipy.ndimage.map_coordinates for fast bilinear interpolation
- Vectorized sampling along the LineString

Dependencies:
- numpy
- shapely
- requests
- rasterio
- scipy
"""

DEFAULT_IMAGE_SERVER_URL = (
    'https://ahn.arcgisonline.nl/arcgis/rest/services/'
    'Hoogtebestand/AHN4_DTM_50cm/ImageServer'
)


class AHN4_API:
    def __init__(self, image_server_url=DEFAULT_IMAGE_SERVER_URL,
                 resolution=1.0, default_buffer=2.5):
        """Create client.

        Args:
            image_server_url: Esri ImageServer REST endpoint
            resolution: requested resolution in meters (default 1.0)
            default_buffer: buffer in meters applied around the LineString to form bbox
        """
        self.base_url = image_server_url.rstrip('/')
        self.resolution = float(resolution)
        self.default_buffer = float(default_buffer)

        self._cache = {}

    # ----------------------------- Cache utilities -----------------------------
    def _bbox_key(self, bbox):
        quant = self.resolution
        q = tuple(round(c / quant) for c in bbox)
        return q

    def clear_cache(self):
        self._cache.clear()

    # ---------------------------- Raster retrieval ----------------------------
    def get_raster_from_wcs(self, bbox, raster=None, force_download=False):
        """Retrieve raster clipped to bbox from Esri ImageServer, with caching.

        Kept as get_raster_from_wcs for backwards compatibility.

        Returns: (data: 2D numpy array, transform: Affine)
        """
        key = self._bbox_key(bbox)
        if (not force_download) and key in self._cache:
            return self._cache[key]

        if bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
            raise ValueError('BBox has zero width/height')

        width = max(1, int(round((bbox[2] - bbox[0]) / self.resolution)))
        height = max(1, int(round((bbox[3] - bbox[1]) / self.resolution)))

        params = {
            'bbox': f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}',
            'bboxSR': 28992,
            'imageSR': 28992,
            'size': f'{width},{height}',
            'format': 'tiff',
            'interpolation': 'RSP_BilinearInterpolation',
            'f': 'image',
        }

        response = requests.get(f'{self.base_url}/exportImage', params=params)
        response.raise_for_status()

        # The server may return a JSON error even with f=image
        content_type = response.headers.get('Content-Type', '')
        if 'json' in content_type:
            error_info = response.json()
            raise RuntimeError(
                f'ImageServer returned error: {error_info.get("error", error_info)}'
            )

        if not RASTERIO_AVAILABLE:
            raise RuntimeError(
                'rasterio is required to decode GeoTIFF from ImageServer'
            )

        try:
            with MemoryFile(response.content) as mem:
                with mem.open() as src:
                    data = src.read(1).astype('float32')
                    transform = src.transform
                    data[data > 9_000] = np.nan
        except Exception as e:
            raise RuntimeError(
                'Failed to read GeoTIFF from ImageServer. Exception: ' + str(e)
            )

        self._cache[key] = (data, transform)
        return data, transform

    # -------------------------- Elevation extraction --------------------------
    def get_elevation_from_line(self, linestring: LineString, spacing=0.5, buffer=None,
                                raster=None, correction=0.0, n_points=None):
        """Sample elevations along a LineString.

        Args:
            linestring: shapely LineString in EPSG:28992 coordinates
            spacing: meters between samples along the line (ignored if n_points provided)
            buffer: buffer in meters to expand bbox around line (default uses self.default_buffer)
            raster: unused, kept for API compatibility
            correction: subtract from L distances (for custom distance origins)
            n_points: if provided, use this many evenly spaced samples (overrides spacing)

        Returns: LineString with Z values (3D)
        """
        if buffer is None:
            buffer = self.default_buffer

        length = linestring.length
        if n_points is not None:
            distances = np.linspace(0, length, n_points)
        else:
            if spacing <= 0:
                raise ValueError('spacing must be > 0')
            n = max(2, int(np.ceil(length / spacing)) + 1)
            distances = np.linspace(0, length, n)

        pts = [linestring.interpolate(d) for d in distances]
        xs = np.array([p.x for p in pts])
        ys = np.array([p.y for p in pts])

        minx, miny, maxx, maxy = linestring.bounds
        bbox = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

        data, transform = self.get_raster_from_wcs(bbox)

        try:
            inv = ~transform
        except Exception:
            x0, y0, x1, y1 = bbox
            resx = (x1 - x0) / float(data.shape[1])
            resy = (y1 - y0) / float(data.shape[0])
            transform = Affine(resx, 0.0, x0, 0.0, -resy, y1)
            inv = ~transform

        cols_rows = [inv * (x, y) for x, y in zip(xs, ys)]
        cols = np.array([c for c, r in cols_rows], dtype=float)
        rows = np.array([r for c, r in cols_rows], dtype=float)

        coords = np.vstack([rows, cols])

        eps = 1e-6
        coords[0] = np.clip(coords[0], -0.5 + eps, data.shape[0] - 0.5 - eps)
        coords[1] = np.clip(coords[1], -0.5 + eps, data.shape[1] - 0.5 - eps)

        Z = map_coordinates(data, coords, order=1, mode='nearest')

        linestring_3d = LineString([(x, y, z_val) for (x, y), z_val in zip(zip(xs, ys), Z)])
        return linestring_3d

    # ---------------------------- Convenience API -----------------------------
    def set_resolution(self, res):
        self.resolution = float(res)
        self.clear_cache()

    def list_coverages(self):
        return ['AHN4_DTM_50cm']
