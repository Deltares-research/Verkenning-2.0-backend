from typing import Union

import numpy as np
from pathlib import Path
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
import geopandas as gpd

from .AHN_raster_API import AHN4_API
from .unit_costs_and_surcharges import load_kosten_catalogus
from .utils import reproject_polygon_with_z


class DikeModel:
    def __init__(self, design_export_3d: gpd.GeoDataFrame, grid_size: float = 0.525):
        self.grid_size = grid_size  # Grid size for area calculations (default 0.525m for ~4070m² match)
        self.design_export_3d = design_export_3d
        self.design_export_3d["geometry"] = self.design_export_3d["geometry"].apply(reproject_polygon_with_z)

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

    def calculate_volume_v3_v4_v5(self, design_3d_surface: gpd.GeoSeries,
                                  thickness_top_layer: float = 0.2,
                                  thickness_clay_layer: float = 0.8) -> tuple[float, float, float]:

        """
        Compute the following volumes:
            - V3: volume of top layer fill (0.2m thick)
            - V4: volume of clay layer fill (0.8m thick)
            - V5: volume of sand layer fill (remaining volume below clay layer and above the current AHN surface)
        """

        clay_layer_top_surface = []
        sand_layer_top_surface = []

        # Create the surfaces for the top of the clay layer and top layer based on the design surface.
        for row in list(design_3d_surface):
            clay_layer_top_surface.append(Polygon([(x, y, z - thickness_top_layer) for x, y, z in row.exterior.coords]))
            sand_layer_top_surface.append(
                Polygon([(x, y, z - thickness_top_layer - thickness_clay_layer) for x, y, z in row.exterior.coords]))

        volume_below_design_surface = self.calculate_volume_below_surface(design_3d_surface).get('fill_volume')
        volume_below_top_layer = self.calculate_volume_below_surface(clay_layer_top_surface).get('fill_volume')
        volume_below_clay_layer = self.calculate_volume_below_surface(sand_layer_top_surface).get('fill_volume')

        V3 = volume_below_design_surface - volume_below_top_layer
        V4 = volume_below_top_layer - volume_below_clay_layer
        V5 = volume_below_clay_layer

        return V3, V4, V5

    def calculate_volume_v1b_v2b(self, design_3d_surface: gpd.GeoSeries, thickness_top_layer: float = 0.2,
                                 thickness_clay_layer: float = 0.8) -> tuple[float, float, float]:
        """
        Compute re-usable volumes:
            - V1b
            - V2b
            - S0: surface area beyond the toe of the old dike
        Assumption is made to determine where the toe location of the old dike is located.
        The volume V1b and V2b are calculated based on the surface area of the current AHN surface, times the thickness of each layers.
        """
        RATIO_TOE_DIKE_TO_EXTENT = 0.3  # It is difficult to locate the toe lijn automatically. The workaround is to assume that the old toe dike is approximately at 30% of the new dike extent.
        combined_poly = unary_union(design_3d_surface)

        grid_pts_global = self.polygon_grid_2d_vectorized(combined_poly, cellsize=self.grid_size)

        elev_global = self.get_elevations(AHN4_API(resolution=self.grid_size), combined_poly, grid_pts_global)

        # Build TIN from valid AHN points
        valid = ~np.isnan(elev_global)
        points_xy = grid_pts_global[valid]
        points_z = elev_global[valid]

        points_3d = np.column_stack((points_xy, points_z))  # (N,3)
        points_xy = points_3d[:, :2]
        tri = Delaunay(points_xy)
        triangles = tri.simplices  # indices of triangle vertices

        def triangle_area(p1, p2, p3):
            return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

        area = 0.0
        for tri_indices in triangles:
            p1, p2, p3 = points_3d[tri_indices]
            area += triangle_area(p1, p2, p3)

        print("Surface area:", area)

        V1b = area * thickness_top_layer * RATIO_TOE_DIKE_TO_EXTENT
        V2b = area * thickness_clay_layer * RATIO_TOE_DIKE_TO_EXTENT
        S0 = area * (1 - RATIO_TOE_DIKE_TO_EXTENT)
        return V1b, V2b, S0

    def calculate_all_dike_volumes(self, thickness_top_layer: float = 0.2, thickness_clay_layer: float = 0.8) -> dict:
        """
        Calculate all dike volumes:
        V1b = volumes['V1b']  # Volume grasbekleding van het huidig profiel (verwijderd en hergebruikt)
        V2b = volumes['V2b']  # Volume kleilaag van het huidig profiel (verwijderd en hergebruikt als kernmateriaal)
        V3 = volumes['V3']  # volume grasbekleding van de nieuwe dijk
        V4 = volumes['V4']  # volume kleilaag van de nieuwe dijk
        V5 = volumes['V5']  # volume kernmateriaal van de nieuwe dijk
        S0 = volumes['S0']  # surface area beyond the toe of the old dike
        """

        design_3d_surface = self.design_export_3d.geometry

        ##### Calculate filling volumes V3, V4, V5:
        V3, V4, V5 = self.calculate_volume_v3_v4_v5(design_3d_surface, thickness_top_layer=thickness_top_layer,
                                                    thickness_clay_layer=thickness_clay_layer)

        #### Calculate re-useable volumes 1b and 2b:
        V1b, V2b, S0 = self.calculate_volume_v1b_v2b(design_3d_surface, thickness_top_layer=thickness_top_layer,
                                                     thickness_clay_layer=thickness_clay_layer)

        return {
            'V1b': V1b,
            'V2b': V2b,
            'V3': V3,
            'V4': V4,
            'V5': V5,
            'S0': S0
        }

    def compute_cost(self, nb_houses: int, road_area: float, complexity: str) -> dict:
        """
        Calculate all the cost for a dike model: groundwork, construction, engineering, real estate costs.

        input:
            nb_houses_intersected: number of houses intersected by the dike (for real estate costs)
            road_area: area of roads to be removed (m²)
            complexity: 'easy', 'medium', 'complex' (for engineering costs with opslagfactor)
        output:
            cost structure dictionary
        """

        path_cost = Path(__file__).parent.joinpath("datasets/eenheidsprijzen.json")
        path_opslag_factor = Path(__file__).parent.joinpath("datasets/opslagfactoren.json")
        cat = load_kosten_catalogus(eenheidsprijzen=str(path_cost), opslagfactoren=str(path_opslag_factor))
        ground_dict = {item.code: item for item in cat.categorieen['Grondverzet']}
        ground_dict_other = {item.code: item for item in cat.categorieen['Profielafwerking']}
        general_construction_dict = {item.code: item for item in cat.categorieen['Algemene werkzaamheden']}
        roads_cost_dict = {item.code: item for item in cat.categorieen['Wegen fietspaden en op-/afritten']}

        Q_GV010 = ground_dict['Q-GV010'].prijs
        Q_GV030 = ground_dict['Q-GV030'].prijs
        Q_GV050 = ground_dict['Q-GV050'].prijs
        Q_GV060 = ground_dict['Q-GV060'].prijs
        Q_GV070 = ground_dict['Q-GV070'].prijs
        Q_GV080 = ground_dict['Q-GV080'].prijs
        Q_GV090 = ground_dict['Q-GV090'].prijs
        Q_GV100 = ground_dict_other['Q-GV100'].prijs
        Q_GV110 = ground_dict_other['Q-GV110'].prijs
        Q_GV120 = ground_dict_other['Q-GV120'].prijs
        Q_AW010 = general_construction_dict['Q-AW010'].prijs
        Q_AW020 = general_construction_dict['Q-AW020'].prijs
        Q_AW030 = general_construction_dict['Q-AW030'].prijs
        ROAD_UNIT_COST = roads_cost_dict['O-413'].prijs + roads_cost_dict['O-513'].prijs

        ### Calculate filling volumes V3, V4, V5:
        THICKNESS_TOP_LAYER = 0.2  # meters
        THICKNESS_CLAY_LAYER = 0.8  # meters
        volumes = self.calculate_all_dike_volumes(THICKNESS_TOP_LAYER, THICKNESS_CLAY_LAYER)
        V1b = volumes['V1b']  # Volume grasbekleding van het huidig profiel (verwijderd en hergebruikt)
        V2b = volumes['V2b']  # Volume kleilaag van het huidig profiel (verwijderd en hergebruikt als kernmateriaal)
        V3 = volumes['V3']  # volume grasbekleding van de nieuwe dijk
        V4 = volumes['V4']  # volume kleilaag van de nieuwe dijk
        V5 = volumes['V5']  # volume kernmateriaal van de nieuwe dijk
        S0 = volumes['S0']  # surface area beyond the toe of the old dike
        S5 = self.calculate_total_3d_surface_area().get(
            'total_3d_area_m2')  # assume S3 = S4 = S5: 3D surface area of new dike

        ### Combine to get costs
        groundwork_cost = (
                S0 * (Q_AW010 + Q_AW020) +  # Voorbereiden terrein
                V1b * Q_GV010 +  # afgraven oude grasbekleding naar depot
                V2b * Q_GV030 +  # afgraven oude kleilaag naar depot
                V2b * Q_GV050 +  # hergebruiken oude kleilaag in nieuwe kern
                (V5 + V1b) * Q_GV090 +  # aanvullen nieuwe kern met nieuw materiaal
                S5 * Q_GV100 +  # profieleren van dijkkern
                V4 * Q_GV080 +  # aanbregen nieuwe kleilaag
                S5 * Q_GV110 +  # profileren nieuwe kleilaar
                V1b * Q_GV060 +  # hergebruiken teelaarde in nieuwe bekleding
                (V3 - V1b) * Q_GV070 +  # aanvullen teelaarde in nieuwe bekleding
                S5 * (Q_GV120 - Q_AW030)  # profileren nieuwe graslaag en inzaaien
        )

        road_removal_cost = road_area * ROAD_UNIT_COST

        return {"Directe bouwkosten": {"Voorbereiding": 0,
                                       "Grondwerk": groundwork_cost,
                                       "Constructie": 0,
                                       },

                "Engineeringkosten": self.calc_engineering_cost(groundwork_cost, complexity, cat),

                "Vastgoedkosten": {"Panden": 0,
                                   "Wegen": road_removal_cost,
                                   },
                }

    def calc_engineering_cost(self, total_direct_cost: float, complexity: str, catalogue)-> dict:
        """
        Calculate the engineering cost as a fraction of the total direct cost, based on the complexity and the price catalogue.

        :param total_direct_cost: Total direct construction cost
        :param complexity: Complexity level ('makkelijke maatregel', 'gemiddelde maatregel', 'moeilijke maatregel')
        :param catalogue: Cost catalogue with opslag factors
        """

        opslag_factor_dict = {item.code: item for item in catalogue.categorieen[
            'Percentages ter bepaling Opslagfactor investeringskosten / benoemde directe bouwkosten algemeen']}
        if complexity == 'makkelijke maatregel':
            engineering_cost_EPK = opslag_factor_dict["Q-ENGOG1"].prijs
            engineering_cost_schets = opslag_factor_dict["Q-ENGON1"].prijs
        elif complexity == 'gemiddelde maatregel':
            engineering_cost_EPK = opslag_factor_dict["Q-ENGOG2"].prijs
            engineering_cost_schets = opslag_factor_dict["Q-ENGON2"].prijs

        elif complexity == 'moeilijke maatregel':
            engineering_cost_EPK = opslag_factor_dict["Q-ENGOG3"].prijs
            engineering_cost_schets = opslag_factor_dict["Q-ENGON3"].prijs

        else:
            raise ValueError(f"Unknown complexity level: {complexity}")
        return {"engineering_cost_EPK": total_direct_cost * engineering_cost_EPK / 100.0,
                "engineering_cost_schets": total_direct_cost * engineering_cost_schets / 100.0}

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
        # Combine polygons to get full extent
        combined_poly = unary_union(surface)

        # 2) Generate a global grid
        grid_pts_global = self.polygon_grid_2d_vectorized(combined_poly, cellsize=self.grid_size)

        # 3) Get the AHN elevations for the grid
        elev_global = self.get_elevations(AHN4_API(resolution=self.grid_size), combined_poly, grid_pts_global)

        nan_count = np.isnan(elev_global).sum()
        valid_count = len(elev_global) - nan_count

        if valid_count == 0:
            print("⚠️  ERROR: NO VALID ELEVATION DATA!")
            return {'fill_volume': 0.0, 'cut_volume': 0.0, 'total_volume': 0.0, 'area': 0.0, 'grid_points': 0}

        # 4) Precompute masks for each polygon
        masks = []
        for row in list(surface):
            poly = row
            path = MplPath(np.array([[x, y] for x, y, *_ in poly.exterior.coords]))
            mask = path.contains_points(grid_pts_global)
            points_in_poly = np.sum(mask)
            masks.append(mask)

        # 5) Compute volumes per polygon WITH INTERPOLATION
        tot_volume_fill, tot_volume_cut = 0.0, 0.0

        print("\n=== VOLUME CALCULATIONS (INTERPOLATED) ===")
        for idx, row in enumerate(list(surface)):
            poly = row
            mask = masks[idx]

            # Get grid points inside this polygon
            grid_pts_in_poly = grid_pts_global[mask]
            elev_poly = elev_global[mask]

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
            fill = np.sum(dV[dV > 0] * self.grid_size ** 2)
            cut = np.sum(-dV[dV < 0] * self.grid_size ** 2)

            tot_volume_fill += fill
            tot_volume_cut += cut

        return {
            'fill_volume': tot_volume_fill,
            'cut_volume': tot_volume_cut,
            'total_volume': tot_volume_fill - tot_volume_cut,
            'area': len(grid_pts_global) * (self.grid_size ** 2),
            'grid_points': len(grid_pts_global)
        }

    def calculate_ruimtebeslag_2d(self, alpha: float = 5.0):
        """
        Calculate the 2D ruimtebeslag (footprint area) where design is above ground.
        
        Creates alpha shape (concave hull) polygons around grid points where design 
        elevation exceeds ground level (AHN).
        
        :param alpha: Buffer size in meters for alpha shape approximation (default 5.0)
        :return: Dictionary with GeoJSON-compatible polygon features and area statistics
        """

        print("\n=== RUIMTEBESLAG 2D CALCULATION (Alpha Shape) ===")

        # 1) Combine polygons to get full extent
        combined_poly = unary_union(self.design_export_3d.geometry)

        # 2) Generate a global grid
        grid_pts_global = self.polygon_grid_2d_vectorized(combined_poly, cellsize=self.grid_size)

        # 3) Get the AHN elevations for the grid
        elev_global = self.get_elevations(AHN4_API(resolution=self.grid_size), combined_poly, grid_pts_global)

        valid_count = len(elev_global) - np.isnan(elev_global).sum()

        if valid_count == 0:
            print("⚠️  ERROR: NO VALID ELEVATION DATA!")
            return {'type': 'FeatureCollection', 'features': [], 'total_area_m2': 0.0, 'num_polygons': 0}

        # 4) Precompute masks for each polygon
        masks = []
        for idx, row in self.design_export_3d.iterrows():
            poly = row.geometry
            path = MplPath(np.array([[x, y] for x, y, *_ in poly.exterior.coords]))
            mask = path.contains_points(grid_pts_global)
            masks.append(mask)

        # 5) Find all points where design elevation > ground elevation
        above_ground_points = []

        for idx, row in self.design_export_3d.iterrows():
            poly = row.geometry
            mask = masks[idx]

            grid_pts_in_poly = grid_pts_global[mask]
            elev_poly = elev_global[mask]

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
            return {'type': 'FeatureCollection', 'features': [], 'total_area_m2': 0.0, 'num_polygons': 0}

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
                'points_above_ground': above_ground_points
            }

        except Exception as e:

            print(f"Error creating alpha shape: {e}")
            import traceback
            traceback.print_exc()
            return {'type': 'FeatureCollection', 'features': [], 'total_area_m2': 0.0, 'num_polygons': 0}

        return {
            'ruimtebeslag_2d_points': above_ground_points
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

    def calculate_3d_surface_area_above_ahn(self, grid_size: float = None):
        """
        Calculate the 3D surface area of planar polygons only where the surface is above AHN.

        :param ahn_raster_func: function that returns AHN elevation at (x, y)
                                e.g., ahn_raster_func(x, y) -> float
        :return: dict with total 3D area
        """
        print("\n=== 3D SURFACE AREA ABOVE AHN (PLANAR POLYGONS) ===")

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
            coords_3d = np.array(row.geometry.exterior.coords)
            # Get AHN elevation at each vertex
            xy_coords = coords_3d[:, :2]
            ahn_elevs = self.get_elevations(AHN4_API(resolution=1.0), row.geometry, xy_coords)
            z_coords = coords_3d[:, 2]

            # Case 1: Entire polygon is above AHN
            if np.all(z_coords > ahn_elevs):
                total_area += planar_polygon_area_3d(coords_3d)
                continue

            # Case 2: Some vertices below AHN -> clip polygon at AHN plane
            clipped_coords = []
            n = len(coords_3d)
            for i in range(n):
                curr = coords_3d[i]
                next_pt = coords_3d[(i + 1) % n]
                curr_z, next_z = curr[2], next_pt[2]
                curr_ahn, next_ahn = ahn_elevs[i], ahn_elevs[(i + 1) % n]

                curr_above = curr_z > curr_ahn
                next_above = next_z > next_ahn

                if curr_above:
                    clipped_coords.append(curr.tolist())

                # Edge crosses AHN plane -> compute intersection
                if curr_above != next_above:
                    # Linear interpolation to intersection point
                    t = (next_ahn - curr_ahn) / ((next_z - next_ahn) - (curr_z - curr_ahn))
                    intersection = curr + t * (next_pt - curr)
                    clipped_coords.append(intersection.tolist())

            if len(clipped_coords) >= 3:
                total_area += planar_polygon_area_3d(clipped_coords)

        return {'total_3d_area_m2': total_area}
