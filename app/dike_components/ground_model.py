from typing import Union, Optional

import numpy as np
from pathlib import Path
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
import geopandas as gpd

from ..AHN_raster_API import AHN4_API, DEFAULT_PDOK_WCS_URL
from ..utils import reproject_polygon_with_z


class GroundModel:
    def __init__(
            self,
            design_export_3d: gpd.GeoDataFrame,
            grid_size: float = 0.525,
            excavation_mode='envelope',
        ahn_source: str = 'arcgis',
        ahn_wcs_url: Optional[str] = None,
    ):
        '''Model representing the ground and associated volume calculations for a dike design. The ground model is initialized with a 3D design surface, and can compute volumes of soil to be excavated or filled based on the difference between the design surface and the current ground surface (from AHN).
        two excavation modes are supported:
        - envelope: the design surface is treated as an envelope. All cells where soil is below the design surface will be supplemented. All cells where soil is above the design surface will be left as is. This is the default method and is preferable for typical soil reinforcements in order to avoid removing all on and off ramps to the dike.
        - cut_and_fill: separate cut and fill volumes are calculated based on the actual difference between the design surface and the AHN surface. In the end the profile will be equal to the design surface on all locations. This approach is more suitable for dike reconstructions where the entire dike profile is removed and rebuilt, such as in a realignment. This options is available but not provided to users at the moment.
        '''
        self.grid_size = grid_size  # Grid size for area calculations (default 0.525m for ~4070m² match)
        self.design_export_3d = design_export_3d
        self.design_export_3d["geometry"] = self.design_export_3d["geometry"].apply(reproject_polygon_with_z)
        if not excavation_mode in ['envelope', 'cut_and_fill']:
            raise ValueError(f"Invalid excavation mode: {excavation_mode}. Must be 'envelope' or 'cut_and_fill'.")
        self.excavation_mode = excavation_mode  # cut_and_fill or envelope.
        self.ahn_source = ahn_source
        self.ahn_wcs_url = ahn_wcs_url

        self.import_elevation_data()

    def _filter_triangles_by_max_edge_length(
            self,
            points_xy: np.ndarray,
            triangles: np.ndarray,
            max_edge_length: Optional[float]
    ) -> np.ndarray:
        """Filter Delaunay triangles by a maximum allowed edge length.

        Triangles are retained only when their longest edge is smaller than or
        equal to ``max_edge_length``. This helps remove unrealistically large
        triangles that can span gaps in sparse or irregular point clouds and
        would otherwise bias area estimates.

        :param points_xy: Array of point coordinates with shape ``(N, 2)``.
            Indices referenced by ``triangles`` must exist in this array.
        :param triangles: Triangle vertex indices with shape ``(M, 3)``,
            typically from ``scipy.spatial.Delaunay(...).simplices``.
        :param max_edge_length: Maximum allowed triangle edge length in meters.
            If ``None`` or ``<= 0``, no filtering is applied.
        :return: Filtered triangle index array with the same column structure
            as input ``triangles``.
        """
        if triangles.size == 0 or max_edge_length is None or max_edge_length <= 0:
            return triangles

        tri_pts = points_xy[triangles]
        edge01 = np.linalg.norm(tri_pts[:, 0, :] - tri_pts[:, 1, :], axis=1)
        edge12 = np.linalg.norm(tri_pts[:, 1, :] - tri_pts[:, 2, :], axis=1)
        edge20 = np.linalg.norm(tri_pts[:, 2, :] - tri_pts[:, 0, :], axis=1)
        max_edge_per_triangle = np.maximum(np.maximum(edge01, edge12), edge20)
        keep_mask = max_edge_per_triangle <= max_edge_length
        return triangles[keep_mask]

    def _filter_horizontal_triangles(
            self,
            points_3d: np.ndarray,
            triangles: np.ndarray,
            min_z_span: float = 1e-6
    ) -> np.ndarray:
        """Remove triangles that are (near-)horizontal in 3D space.

        A triangle is considered horizontal when the spread in its vertex Z
        values is smaller than or equal to ``min_z_span``.

        :param points_3d: Array of XYZ coordinates with shape ``(N, 3)``.
        :param triangles: Triangle vertex indices with shape ``(M, 3)``.
        :param min_z_span: Minimum required ``max(z) - min(z)`` in meters for
            a triangle to be kept.
        :return: Filtered triangle index array.
        """
        if triangles.size == 0:
            return triangles

        tri_pts = points_3d[triangles]
        z_span = np.ptp(tri_pts[:, :, 2], axis=1)
        keep_mask = z_span > min_z_span
        return triangles[keep_mask]

    def import_elevation_data(self):
        # Combine polygons to get full extent
        combined_poly = unary_union(self.design_export_3d["geometry"])

        # 2) Generate a global grid
        self.grid_pts_global = self.polygon_grid_2d_vectorized(combined_poly, cellsize=self.grid_size)

        # 3) Get the AHN elevations for the grid
        ahn_client = AHN4_API(
            resolution=self.grid_size,
            source=self.ahn_source,
            wcs_url=self.ahn_wcs_url if self.ahn_wcs_url else DEFAULT_PDOK_WCS_URL,
        )
        self.elev_global = self.get_elevations(ahn_client, combined_poly, self.grid_pts_global)

    def polygon_grid_2d_vectorized(self, poly: Polygon, cellsize: float = 1.0) -> np.ndarray:
        """Generate grid points inside polygon using fully vectorized operations.

        :param poly: Shapely Polygon (2D)
        :param cellsize: grid cell size
        :return: Nx2 array of points inside polygon
        """
        minx, miny, maxx, maxy = poly.bounds
        x = np.arange(minx + 0.5 * cellsize, maxx, cellsize)
        y = np.arange(miny + 0.5 * cellsize, maxy, cellsize)
        xx, yy = np.meshgrid(x, y)
        points = np.vstack((xx.ravel(), yy.ravel())).T

        # extract exterior coordinates, ignoring Z
        poly_coords = np.array([[px, py] for px, py, *_ in poly.exterior.coords])
        path = MplPath(poly_coords)

        mask = path.contains_points(points)
        self.grid_2d = points[mask]
        return points[mask]

    def get_elevations(self, ahn_client: AHN4_API, polygon: Polygon, grid_points: np.ndarray) -> np.ndarray:
        """
        Get elevations from AHN raster for given grid points within polygon.
        :param ahn_client: AHN4_API client
        :param polygon:
        :param grid_points:
        :return:
        """
        minx, miny, maxx, maxy = polygon.bounds
        data, transform = ahn_client.get_raster_from_wcs((minx, miny, maxx, maxy))
        inv = ~transform
        cols_rows = np.array([inv * (x, y) for x, y in grid_points])
        rows = cols_rows[:, 1]
        cols = cols_rows[:, 0]
        # coords = np.vstack([cols, rows])  # note: check order with map_coordinates
        coords = np.vstack([rows, cols])
        coords[0] = np.clip(coords[0], 0, data.shape[0] - 1)
        coords[1] = np.clip(coords[1], 0, data.shape[1] - 1)
        elev = map_coordinates(data, coords, order=1)
        mean_z = np.nanmean(elev)
        std_z = np.nanstd(elev)
        mask = np.abs(elev - mean_z) < 3 * std_z
        elev_filtered = elev.copy()
        elev_filtered[~mask] = mean_z  # replace spikes with mean

        self.elevation = elev
        return elev

    def calculate_volume_v3_v4_v5(self,
                                  thickness_top_layer: float = 0.2,
                                  thickness_clay_layer: float = 0.8,
                                  ) -> tuple[float, float, float]:

        """
        Compute the following volumes:
            - V3: volume of top layer fill (0.2m thick)
            - V4: volume of clay layer fill (0.8m thick)
            - V5: volume of sand layer fill (remaining volume below clay layer and above the current AHN surface)

        The computed volumes are made for the envelop of the design!! Because the method self.calculate_volume_below_surface
        applies a mask where AHN > design
        """

        clay_layer_top_surface = []
        sand_layer_top_surface = []

        # Create the surfaces for the top of the clay layer and top layer based on the design surface.
        for row in list(self.design_export_3d.geometry):
            coords = np.array(row.exterior.coords)
            xy = coords[:, :2]
            z_vals = coords[:, 2]

            clay_layer_top_surface.append(
                Polygon([(x, y, z - thickness_top_layer) for (x, y), z in zip(xy, z_vals)])
            )
            sand_layer_top_surface.append(
                Polygon([
                    (x, y, z - thickness_top_layer - thickness_clay_layer)
                    for (x, y), z in zip(xy, z_vals)
                ])
            )

        volume_below_design_surface = self.calculate_volume_below_surface(self.design_export_3d.geometry).get(
            'fill_volume')
        volume_below_top_layer = self.calculate_volume_below_surface(clay_layer_top_surface).get('fill_volume')
        volume_below_clay_layer = self.calculate_volume_below_surface(sand_layer_top_surface).get('fill_volume')

        V3 = volume_below_design_surface - volume_below_top_layer
        V4 = volume_below_top_layer - volume_below_clay_layer
        V5 = volume_below_clay_layer

        return V3, V4, V5

    def calculate_volume_v1b_v2b(self, area_ahn_profile: float, thickness_top_layer: float = 0.2,
                                 thickness_clay_layer: float = 0.8) -> tuple[float, float]:
        """
        Compute re-usable volumes:
            - V1b
            - V2b
            - S0: surface area where the reinforcement will take place
        Assumption is made to determine where the toe location of the old dike is located.
        The volume V1b and V2b are calculated based on the surface area of the current AHN surface, times the thickness of each layers.
        """
        # Currently we assume that the entire area where there will be works is identical and has to be cleared and excavated in the same way. A future improvement could be to distinguish the part that is on the existing dike (with clear top, clay layers) and the part that is behind it

        V1b = area_ahn_profile * thickness_top_layer
        V2b = area_ahn_profile * thickness_clay_layer
        return V1b, V2b

    def _interpolate_design_heights_on_global_grid(self) -> np.ndarray:
        """
        Interpolate design-surface Z-values on the global XY grid.

        :return: 1D array of interpolated design heights (NaN where unavailable)
        """
        design_z_global = np.full(len(self.grid_pts_global), np.nan, dtype=float)

        for poly in list(self.design_export_3d.geometry):
            path = MplPath(np.array([[x, y] for x, y, *_ in poly.exterior.coords]))
            mask = path.contains_points(self.grid_pts_global)
            if not np.any(mask):
                continue

            poly_coords_3d = np.array(poly.exterior.coords)
            interpolated_z = griddata(
                points=poly_coords_3d[:, :2],
                values=poly_coords_3d[:, 2],
                xi=self.grid_pts_global[mask],
                method='linear',
                fill_value=np.mean(poly_coords_3d[:, 2])
            )
            design_z_global[mask] = interpolated_z

        return design_z_global

    def calculate_3d_surface_TIN(
            self,
            height_source: str = 'ahn',
            exclude_points_where_ahn_above_design: bool = False
    ) -> dict:
        """
        Calculate the 3d surface area using a TIN method. This method uses a special interpolation technique to
        compute area for a non-planar surface. It is computationally more expensive however.

        :param height_source: Source used to build TIN points:
            - ``'ahn'``: XY from global grid with Z from AHN elevations
                        - ``'design'``: XY from global grid with Z from interpolated
                            design heights
            - ``'design_edges'``: XYZ directly from design polygon edge vertices
            - ``design`` : XYZ taken from global grid using interpolated design heights.
        :param exclude_points_where_ahn_above_design: Used when
            ``height_source`` is ``'design'`` or ``'ahn'``. If ``True``,
            excludes grid points where AHN elevation is higher than
            interpolated design elevation.
        :return: Dictionary with ``area`` (float, m²), ``triangles``
            (``np.ndarray`` with shape ``(M, 3)``), and ``points_3d``
            (``np.ndarray`` with shape ``(N, 3)``).
        """

        # Build TIN from valid AHN points
        if height_source not in ['ahn', 'design', 'design_edges']:
            raise ValueError(
                f"Invalid height_source: {height_source}. Must be 'ahn', 'design' or 'design_edges'."
            )

        if height_source == 'design_edges':
            edge_points_3d = []
            for poly in list(self.design_export_3d.geometry):
                coords_3d = np.array(poly.exterior.coords)
                if len(coords_3d) > 1 and np.allclose(coords_3d[0], coords_3d[-1]):
                    coords_3d = coords_3d[:-1]
                edge_points_3d.extend(coords_3d.tolist())

            if len(edge_points_3d) < 3:
                return {
                    'area': 0.0,
                    'triangles': np.empty((0, 3), dtype=int),
                    'points_3d': np.empty((0, 3), dtype=float)
                }

            points_3d = np.array(edge_points_3d, dtype=float)
        elif height_source == 'ahn':
            z_values = self.elev_global
            valid = ~np.isnan(z_values)

            if exclude_points_where_ahn_above_design:
                design_z_values = self._interpolate_design_heights_on_global_grid()
                design_valid = ~np.isnan(design_z_values)
                valid = valid & design_valid & (z_values <= design_z_values)

            points_xy = self.grid_pts_global[valid]
            points_z = z_values[valid]
            if len(points_xy) < 3:
                return {
                    'area': 0.0,
                    'triangles': np.empty((0, 3), dtype=int),
                    'points_3d': np.empty((0, 3), dtype=float)
                }
            points_3d = np.column_stack((points_xy, points_z))  # (N,3)
        else:  # design on global XY grid
            z_values = self._interpolate_design_heights_on_global_grid()
            valid = ~np.isnan(z_values)

            if exclude_points_where_ahn_above_design:
                ahn_valid = ~np.isnan(self.elev_global)
                valid = valid & ahn_valid & (self.elev_global <= z_values)

            points_xy = self.grid_pts_global[valid]
            points_z = z_values[valid]
            if len(points_xy) < 3:
                return {
                    'area': 0.0,
                    'triangles': np.empty((0, 3), dtype=int),
                    'points_3d': np.empty((0, 3), dtype=float)
                }
            points_3d = np.column_stack((points_xy, points_z))  # (N,3)

        points_xy = points_3d[:, :2]
        tri = Delaunay(points_xy)
        triangles = tri.simplices  # indices of triangle vertices

        if height_source == 'design_edges':
            triangles = self._filter_horizontal_triangles(points_3d, triangles)
        elif height_source in ['ahn', 'design']:
            max_edge_length = 8 * self.grid_size  # is the largest edge is 8 times bigger than the grid size, we consider it an unrealistic triangle that spans a gap in the data, and we remove it from the area calculation.
            triangles = self._filter_triangles_by_max_edge_length(points_xy, triangles, max_edge_length)

        if triangles.size == 0:
            area = 0.0
        else:
            tri_pts = points_3d[triangles]  # (M, 3, 3)
            cross = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
            area = 0.5 * np.sum(np.linalg.norm(cross, axis=1))

        print("Surface area:", area)
        return {
            'area': area,
            'triangles': triangles,
            'points_3d': points_3d
        }

    def calculate_all_dike_volumes(self, thickness_top_layer: float = 0.2, thickness_clay_layer: float = 0.8) -> dict:
        """
        Calculate all dike volumes:
        V1b = volumes['V1b']  # Volume grasbekleding van het huidig profiel (verwijderd en hergebruikt)
        V2b = volumes['V2b']  # Volume kleilaag van het huidig profiel (verwijderd en hergebruikt als kernmateriaal)
        V3 = volumes['V3']  # volume grasbekleding van de nieuwe dijk
        V4 = volumes['V4']  # volume kleilaag van de nieuwe dijk
        V5 = volumes['V5']  # volume kernmateriaal van de nieuwe dijk


        """

        full_AHN_surface = self.calculate_3d_surface_TIN(height_source='ahn')[
            'area']  # This is the area of the 3D surface of the AHN profile under the entire design footprint

        envelop_AHN_surface = self.calculate_3d_surface_TIN(height_source='ahn',
                                                            exclude_points_where_ahn_above_design=True)['area']  # this is the area of the 3D surface of the AHN profile where AHN is below the design surface

        full_design_surface = self.calculate_total_3d_surface_area().get(
            'total_3d_area_m2')  # assume S3 = S4 = S5: 3D surface area of new dike

        #### Calculate re-useable volumes 1b and 2b:
        V1b, V2b = self.calculate_volume_v1b_v2b(thickness_top_layer=thickness_top_layer,
                                                 thickness_clay_layer=thickness_clay_layer,
                                                 area_ahn_profile=envelop_AHN_surface
                                                 )

        #### Calculate V3, V4, V5 based on Envelop design
        envelop_design_surface = self.calculate_3d_surface_TIN(height_source='design', exclude_points_where_ahn_above_design=True)['area']
        V3, V4, V5 = self.calculate_volume_v3_v4_v5(
            thickness_top_layer=thickness_top_layer,
            thickness_clay_layer=thickness_clay_layer)

        return {
            'V1b': V1b,
            'V2b': V2b,
            'V3': V3,
            'V4': V4,
            'V5': V5,
            'full_AHN_surface': full_AHN_surface,
            'envelop_AHN_surface': envelop_AHN_surface,
            'full_design_surface': full_design_surface,
            'envelop_design_surface': envelop_design_surface
        }

    def calculate_volume(self):
        """
        Compute fill and excavation volumes for all polygons using a single global grid
        and single AHN raster query.
        
        ✅ FIXED: Now interpolates design heights at each grid point instead of using mean!
        """

        return self.calculate_volume_below_surface(self.design_export_3d.geometry)

    def calculate_volume_below_surface(self, surface: Union[gpd.GeoSeries, list[Polygon]]):
        """
        Calculate the volume of soil between a designated surface and the AHN ground surface
        """

        nan_count = np.isnan(self.elev_global).sum()
        valid_count = len(self.elev_global) - nan_count

        if valid_count == 0:
            print("⚠️  ERROR: NO VALID ELEVATION DATA!")
            return {'fill_volume': 0.0, 'cut_volume': 0.0, 'net_volume': 0.0, 'area': 0.0, 'grid_points': 0}

        # 4) Precompute masks for each polygon
        masks = []
        for row in list(surface):
            poly = row
            path = MplPath(np.array([[x, y] for x, y, *_ in poly.exterior.coords]))
            mask = path.contains_points(self.grid_pts_global)
            points_in_poly = np.sum(mask)
            masks.append(mask)

        # 5) Compute volumes per polygon WITH INTERPOLATION
        tot_volume_fill, tot_volume_cut = 0.0, 0.0

        print("\n=== VOLUME CALCULATIONS (INTERPOLATED) ===")
        for idx, row in enumerate(list(surface)):
            poly = row
            mask = masks[idx]

            # Get grid points inside this polygon
            grid_pts_in_poly = self.grid_pts_global[mask]
            elev_poly = self.elev_global[mask]

            if len(grid_pts_in_poly) == 0:
                print(f"\nPolygon {idx}: No grid points, skipping")
                continue

            # ✅ FIX: Interpolate design heights at each grid point
            poly_coords_3d = np.array(poly.exterior.coords)

            # Use scipy griddata to interpolate Z at each grid point
            design_heights = griddata(
                points=poly_coords_3d[:, :2],  # XY coordinates of polygon vertices
                values=poly_coords_3d[:, 2],  # Z values at vertices
                xi=grid_pts_in_poly,  # Grid points where to evaluate
                method='linear',  # Linear interpolation
                fill_value=np.mean(poly_coords_3d[:, 2])  # Fallback for points outside convex hull
            )

            # Filter valid AHN elevations
            valid_mask = ~np.isnan(elev_poly)
            design_heights_valid = design_heights[valid_mask]
            elev_poly_valid = elev_poly[valid_mask]

            # Compute volume with interpolated heights
            dV = design_heights_valid - elev_poly_valid
            if self.excavation_mode == 'cut_and_fill':
                fill, cut = self.cut_and_fill_volume_and_area(dV)
            elif self.excavation_mode == 'envelope':
                fill, cut = self.get_envelope_volume_and_area(dV)
            else:
                raise ValueError(f"Invalid excavation mode: {self.excavation_mode}")
            tot_volume_fill += fill
            tot_volume_cut += cut

        return {
            'fill_volume': tot_volume_fill,
            'cut_volume': tot_volume_cut,
            'net_volume': tot_volume_fill - tot_volume_cut,
            'area': len(self.grid_pts_global) * (self.grid_size ** 2),
            'grid_points': len(self.grid_pts_global)
        }

    def cut_and_fill_volume_and_area(self, dV: np.ndarray) -> dict:
        fill = np.sum(dV[dV > 0] * self.grid_size ** 2)
        cut = np.sum(-dV[dV < 0] * self.grid_size ** 2)
        return fill, cut

    def get_envelope_volume_and_area(self, dV: np.ndarray) -> dict:
        fill = np.sum(np.maximum(dV, 0) * self.grid_size ** 2)
        cut = 0.0  # No excavation in envelope mode
        return fill, cut

    def calculate_ruimtebeslag_2d(self, alpha: float = 5.0):
        """
        Calculate the 2D ruimtebeslag (footprint area) where design is above ground.
        
        Creates alpha shape (concave hull) polygons around grid points where design 
        elevation exceeds ground level (AHN).
        
        :param alpha: Buffer size in meters for alpha shape approximation (default 5.0)
        :return: Dictionary with GeoJSON-compatible polygon features and area statistics
        """

        print("\n=== RUIMTEBESLAG 2D CALCULATION (Alpha Shape) ===")

        valid_count = len(self.elev_global) - np.isnan(self.elev_global).sum()

        if valid_count == 0:
            print("⚠️  ERROR: NO VALID ELEVATION DATA!")
            return {
                'type': 'FeatureCollection',
                'features': [],
                'total_area_m2': 0.0,
                'num_polygons': 0,
                'polygons_rd': []
            }

        # 4) Precompute masks for each polygon
        masks = []
        for idx, row in self.design_export_3d.iterrows():
            poly = row.geometry
            path = MplPath(np.array([[x, y] for x, y, *_ in poly.exterior.coords]))
            mask = path.contains_points(self.grid_pts_global)
            masks.append(mask)

        # 5) Find all points where design elevation > ground elevation
        above_ground_points = []

        for idx, row in self.design_export_3d.iterrows():
            poly = row.geometry
            mask = masks[idx]

            grid_pts_in_poly = self.grid_pts_global[mask]
            elev_poly = self.elev_global[mask]

            if len(grid_pts_in_poly) == 0:
                continue

            # Interpolate design heights at each grid point
            poly_coords_3d = np.array(poly.exterior.coords)
            design_heights = griddata(
                points=poly_coords_3d[:, :2],
                values=poly_coords_3d[:, 2],
                xi=grid_pts_in_poly,
                method='linear',
                fill_value=np.mean(poly_coords_3d[:, 2])
            )

            # Filter for points above ground
            valid_mask = ~np.isnan(elev_poly)
            above_ground_mask = (design_heights[valid_mask] > elev_poly[valid_mask])

            # Collect points that are above ground **with z coordinate**
            pts_xy = grid_pts_in_poly[valid_mask][above_ground_mask]
            z_vals = design_heights[valid_mask][above_ground_mask]

            # Combine into Nx3 array and append
            points_above_3d = np.column_stack([pts_xy, z_vals])
            above_ground_points.extend(points_above_3d.tolist())

        total_area = len(above_ground_points) * (self.grid_size ** 2)

        if len(above_ground_points) < 3:
            print("⚠️  Not enough points above ground to create polygon")
            return {
                'type': 'FeatureCollection',
                'features': [],
                'total_area_m2': 0.0,
                'num_polygons': 0,
                'polygons_rd': []
            }

        # 6) Create alpha shape (concave hull) around above-ground points
        try:
            # We rasterize the points and use morphological operations to approximate alpha shape, this is MUCH faster
            # than using alphashape libraries directly!
            pts = np.array(above_ground_points)
            xmin, ymin, _ = pts.min(axis=0)
            xmax, ymax, _ = pts.max(axis=0)

            # 1m grid
            grid_w = int(np.ceil(xmax - xmin)) + 3
            grid_h = int(np.ceil(ymax - ymin)) + 3

            grid = np.zeros((grid_h, grid_w), dtype=bool)
            grid[(pts[:, 1] - ymin).astype(int), (pts[:, 0] - xmin).astype(int)] = True

            # Morphological dilation + erosion to mimic alpha shape
            from scipy.ndimage import binary_dilation, binary_erosion
            grid = binary_dilation(grid, iterations=2)
            grid = binary_erosion(grid, iterations=2)

            # Extract polygons
            polygons = []
            from skimage import measure
            contours = measure.find_contours(grid.astype(float), 0.5)
            for c in contours:
                c[:, 0] += ymin  # row → y
                c[:, 1] += xmin  # col → x
                poly = Polygon(c[:, [1, 0]])  # (x, y)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)

            # 7) Convert to GeoJSON format (WGS84) and calculate areas
            from pyproj import Transformer
            transformer_to_wgs = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)
            features = []

            for i, poly in enumerate(polygons):
                area_m2 = poly.area

                # Convert coordinates from RD to WGS84
                coords_rd = list(poly.exterior.coords)
                coords_wgs = [list(transformer_to_wgs.transform(x, y)) for x, y in coords_rd]

                feature = {

                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [coords_wgs]
                    },
                    'properties': {
                        'area_m2': round(area_m2, 2),
                        'polygon_id': i
                    }

                }

                features.append(feature)

            return {
                'type': 'FeatureCollection',
                'features': features,
                'crs': {
                    'type': 'name',
                    'properties': {'name': 'EPSG:4326'}
                },
                'total_area_m2': round(total_area, 2),
                'num_polygons': len(features),
                'polygons_rd': polygons,
                'points_above_ground': above_ground_points
            }

        except Exception as e:

            print(f"Error creating alpha shape: {e}")
            import traceback
            traceback.print_exc()
            return {
                'type': 'FeatureCollection',
                'features': [],
                'total_area_m2': 0.0,
                'num_polygons': 0,
                'polygons_rd': []
            }

    def calculate_total_3d_surface_area(self):
        """
        Calculate total 3D surface area assuming every polygon is planar.
        Faster and more accurate than triangulation.

        :return: Dict with total 3D surface area in m²
        """
        print("\n=== TOTAL 3D SURFACE AREA (PLANAR POLYGONS) ===")

        def planar_polygon_area_3d(coords):
            """Compute 3D area of a planar polygon using Newell's method"""
            coords = np.array(coords)
            n = len(coords)
            Ax = Ay = Az = 0.0
            for i in range(n):
                x0, y0, z0 = coords[i]
                x1, y1, z1 = coords[(i + 1) % n]
                Ax += (y0 - y1) * (z0 + z1)
                Ay += (z0 - z1) * (x0 + x1)
                Az += (x0 - x1) * (y0 + y1)
            return 0.5 * np.sqrt(Ax ** 2 + Ay ** 2 + Az ** 2)

        total_area = 0.0
        for idx, row in self.design_export_3d.iterrows():
            coords_3d = row.geometry.exterior.coords
            total_area += planar_polygon_area_3d(coords_3d)

        return {'total_3d_area_m2': total_area}
