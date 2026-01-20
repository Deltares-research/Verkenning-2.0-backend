
"""
Optimized AHN4 client for extracting elevation profiles along LineStrings.

Features:
- Raster caching to avoid repeated WCS downloads
- Small bbox computed from a LineString with configurable buffer
- Uses rasterio (fast GeoTIFF decoding) with fallback to PIL
- Uses scipy.ndimage.map_coordinates for fast bilinear interpolation
- Vectorized sampling along the LineString

Dependencies:
- numpy
- shapely
- owslib
- rasterio (recommended)
- scipy

Example usage at the bottom.
"""

from owslib.wcs import WebCoverageService
from shapely.geometry import LineString
from io import BytesIO
import numpy as np
import time
import warnings

from PIL import Image

from rasterio.io import MemoryFile
from rasterio.transform import Affine
RASTERIO_AVAILABLE = True


from scipy.ndimage import map_coordinates


class AHN4_API:
    def __init__(self, wcs_url='https://service.pdok.nl/rws/ahn/wcs/v1_0?SERVICE=WCS',
                 resolution=1.0, default_buffer=2.5):
        """Create client.

        Args:
            wcs_url: WCS endpoint
            resolution: requested resolution in meters (default 1.0)
            default_buffer: buffer in meters applied around the LineString to form bbox
        """
        self.wcs = WebCoverageService(wcs_url, version='1.0.0')
        # self.coverage_ids = list(self.wcs.contents)
        self.coverage_ids = ['dtm_05m']
        self.resolution = float(resolution)
        self.default_buffer = float(default_buffer)

        self._cache = {}

    # ----------------------------- Cache utilities -----------------------------
    def _bbox_key(self, bbox, raster):
        # quantize bbox to resolution to reduce cache misses
        quant = self.resolution
        q = tuple([round(c / quant) for c in bbox])
        return (raster, q)

    def clear_cache(self):
        self._cache.clear()

    # ---------------------------- Raster retrieval ----------------------------
    def get_raster_from_wcs(self, bbox, raster=None, force_download=False):
        """Retrieve raster clipped to bbox from WCS, with caching.

        Returns: (data: 2D numpy array, transform: Affine)
        """
        if raster is None:
            raster = self.coverage_ids[0]
        elif isinstance(raster, int):
            raster = self.coverage_ids[raster]

        key = self._bbox_key(bbox, raster)
        if (not force_download) and key in self._cache:
            return self._cache[key]

        # Validate bbox
        if bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
            raise ValueError('BBox has zero width/height')

        output = self.wcs.getCoverage(identifier=raster,
                                      bbox=bbox,
                                      resx=self.resolution,
                                      resy=self.resolution,
                                      format='GeoTIFF',
                                      crs='EPSG:28992',
                                      interpolation='AVERAGE')

        content = output.read()

        if RASTERIO_AVAILABLE:
            try:
                with MemoryFile(content) as mem:
                    with mem.open() as src:
                        data = src.read(1)
                        transform = src.transform
                        # replace large invalid values with NaN
                        data = data.astype('float32')
                        data[data > 9_000] = np.nan
            except Exception:
                warnings.warn('rasterio failed to read GeoTIFF, falling back to PIL')
                RASTERIO_FALLBACK = True
                RASTERIO_FALLBACK
        else:
            RASTERIO_FALLBACK = True

        if not RASTERIO_AVAILABLE or 'RASTERIO_FALLBACK' in locals():
            # Fallback using PIL -- slower and lossy for float data
            im = Image.open(BytesIO(content))
            data = np.array(im)
            data = data.astype('float32')
            data[data > 9_000] = np.nan

            # Build a transform: assume data.shape maps to bbox with top-left origin
            x0, y0, x1, y1 = bbox
            height, width = data.shape
            resx = (x1 - x0) / float(width)
            resy = (y1 - y0) / float(height)
            # Affine(c, a, b, f, d, e) but rasterio uses Affine(resx, 0, x0, 0, -resy, y1)
            transform = Affine(resx, 0.0, x0, 0.0, -resy, y1)

        # Store in cache and return
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
            raster: raster id or index for WCS (None => first available)
            correction: subtract from L distances (for custom distance origins)
            n_points: if provided, use this many evenly spaced samples (overrides spacing)

        Returns: (L: 1D distances array, Z: 1D elevations array)
        """
        if buffer is None:
            buffer = self.default_buffer

        # Decide sample distances along the line
        length = linestring.length
        if n_points is not None:
            distances = np.linspace(0, length, n_points)
        else:
            if spacing <= 0:
                raise ValueError('spacing must be > 0')
            n = max(2, int(np.ceil(length / spacing)) + 1)
            distances = np.linspace(0, length, n)

        # Sample coordinates along the line
        pts = [linestring.interpolate(d) for d in distances]
        xs = np.array([p.x for p in pts])
        ys = np.array([p.y for p in pts])

        # small bbox around entire LineString to minimize raster size
        minx, miny, maxx, maxy = linestring.bounds
        bbox = (minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)

        data, transform = self.get_raster_from_wcs(bbox, raster=raster)

        # compute fractional row/col indices for each sample point using inverse transform
        try:
            inv = ~transform
        except Exception:
            # If transform is not Affine (e.g., when PIL fallback made a custom Affine), construct one
            # This is an unlikely path; if it happens, assume regular grid
            x0, y0, x1, y1 = bbox
            resx = (x1 - x0) / float(data.shape[1])
            resy = (y1 - y0) / float(data.shape[0])
            transform = Affine(resx, 0.0, x0, 0.0, -resy, y1)
            inv = ~transform

        cols_rows = [inv * (x, y) for x, y in zip(xs, ys)]
        cols = np.array([c for c, r in cols_rows], dtype=float)
        rows = np.array([r for c, r in cols_rows], dtype=float)

        # map_coordinates expects coords in order [rows, cols]
        coords = np.vstack([rows, cols])

        # clamp coords slightly inside array to avoid exact-edge indexing issues
        eps = 1e-6
        coords[0] = np.clip(coords[0], -0.5 + eps, data.shape[0] - 0.5 - eps)
        coords[1] = np.clip(coords[1], -0.5 + eps, data.shape[1] - 0.5 - eps)

        # perform fast bilinear interpolation (order=1)
        Z = map_coordinates(data, coords, order=1, mode='nearest')

        # Convert distances
        L = distances - correction

        return L, Z

    # ---------------------------- Convenience API -----------------------------
    def set_resolution(self, res):
        self.resolution = float(res)
        self.clear_cache()

    def list_coverages(self):
        return self.coverage_ids

