import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates, median_filter
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

import geopandas as gpd

from backend.AHN_raster_API import AHN4_API


transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
transformer_rd_to_wm = Transformer.from_crs("EPSG:28992", "EPSG:3857", always_xy=True)


def reproject_polygon_with_z(poly: Polygon) -> Polygon:
    """
    Reproject a Polygon Z from WGS84 to RD while keeping the Z coordinate.
    Returns a Polygon with 3D coordinates (x_RD, y_RD, z).
    """
    exterior_coords_3d = []
    for x, y, *z in poly.exterior.coords:
        z_val = z[0] if z else 0.0
        x_rd, y_rd = transformer.transform(x, y)
        exterior_coords_3d.append((x_rd, y_rd, z_val))
    return Polygon(exterior_coords_3d)


class DikeModel:
    # in initialization, pass the design_export_3d GeoDataFrame
    grid_size: float
    design_export_3d: gpd.GeoDataFrame
    executionVolume: float
    fillVolume: float
    totalVolumeDifference: float

    # set attributes during calculation
    elevation: np.ndarray
    grid_2d: np.ndarray

    def __init__(self, design_export_3d: gpd.GeoDataFrame):
        self.grid_size = 1.0
        self.design_export_3d = design_export_3d
        self.design_export_3d["geometry"] = self.design_export_3d["geometry"].apply(reproject_polygon_with_z)

        self.excavationVolume = 0
        self.fillVolume = 0
        self.totalVolumeDifference = 0


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
        path = Path(poly_coords)

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

    def compute_volumes(self, current_elev: np.array, new_elev: np.array, cell_area=1.0) -> tuple[float, float]:
        """

        :param current_elev: array of current elevations (AHN)
        :param new_elev: array of new design elevations (from GIS polygons)
        :param cell_area: grid size
        :return:
        """
        dV = new_elev - current_elev
        fill = np.sum(dV[dV > 0] * cell_area)
        cut = np.sum(-dV[dV < 0] * cell_area)
        return fill, cut

    def calculate_volume(self):
        """
        Compute fill and excavation volumes for all polygons using a single global grid
        and single AHN raster query.

        Parameters:
        - ahn_api: AHN API client
        - cellsize: grid resolution in meters (defaults to self.gridSize)
        """

        # 1) Combine polygons to get full extent
        combined_poly = unary_union(self.design_export_3d.geometry)

        # 2) Generate a global grid (1x1 m or self.grid_size)
        time1 = time.time()
        grid_pts_global = self.polygon_grid_2d_vectorized(combined_poly, cellsize=self.grid_size)
        time2 = time.time()
        print(f"Global grid generated: {len(grid_pts_global)} points in {time2 - time1:.3f}s")

        # 3) Get the AHN elevations for the grid
        time3 = time.time()
        elev_global = self.get_elevations(AHN4_API(resolution=self.grid_size), combined_poly, grid_pts_global)
        time4 = time.time()
        print(f"AHN elevations sampled in {time4 - time3:.3f}s")

        # 4) Precompute masks for each polygon
        masks = []
        for _, row in self.design_export_3d.iterrows():
            poly = row.geometry
            path = Path(np.array([[x, y] for x, y, *_ in poly.exterior.coords]))
            mask = path.contains_points(grid_pts_global)
            masks.append(mask)

        # 5) Compute volumes per polygon
        tot_volume_fill, tot_volume_cut = 0.0, 0.0

        for idx, row in self.design_export_3d.iterrows():

            poly = row.geometry
            #TODO: improve this as it will lead to approximation. Ideally the dike polygon should also be discretized at grid points

            # opt 1: use mean height of the dike profile
            new_height = np.mean([coord[2] for coord in poly.exterior.coords])
            ## opt 2: interpolate heights at grid points
            # poly_coords_3d = np.array(poly.exterior.coords)  # shape: (M, 3)
            # Use griddata to interpolate Z at each grid point
            # new_height = griddata(
            #     points=poly_coords_3d[:, :2],  # XY
            #     values=poly_coords_3d[:, 2],  # Z
            #     xi=grid_pts_global, # points where to evaluate
            #     method='linear'  # linear interpolation
            # )

            mask = masks[idx]

            elev_poly = elev_global[mask]

            fill, cut = self.compute_volumes(elev_poly, new_height, cell_area=self.grid_size ** 2)
            print(row['name'], f'fill (m3): {fill:.2f}, cut (m3): {cut:.2f}')

            tot_volume_fill += fill
            tot_volume_cut += cut

        print(f"Total fill (m3): {tot_volume_fill:.2f}, Total cut (m3): {tot_volume_cut:.2f}")
        
        return {
            'fill_volume': tot_volume_fill,
            'cut_volume': tot_volume_cut,
            'total_volume': tot_volume_fill - tot_volume_cut,
            'area': len(grid_pts_global) * (self.grid_size ** 2),
            'grid_points': len(grid_pts_global)
        }


    def plot_existing_and_new_surface_plotly(
        self,
        grid_resolution=200,
        title="Existing vs New Dike Surface"
    ):
        """
        Dynamic 3D surface plot using Plotly for AHN and new design surfaces.
        """

        # -----------------------------
        # 1. Prepare AHN surface
        # -----------------------------
        grid_points = self.grid_2d
        current_elev = self.elevation
        design_polygons = self.design_export_3d["geometry"].tolist()

        X = grid_points[:, 0]
        Y = grid_points[:, 1]
        Z = current_elev

        xi = np.linspace(X.min(), X.max(), grid_resolution)
        yi = np.linspace(Y.min(), Y.max(), grid_resolution)
        XI, YI = np.meshgrid(xi, yi)

        ZI_ahn = griddata((X, Y), Z, (XI, YI), method='linear')

        # -----------------------------
        # 2. Prepare new design surface
        # -----------------------------
        Xn, Yn, Zn = [], [], []
        for poly in design_polygons:
            for x, y, z in poly.exterior.coords:
                Xn.append(x)
                Yn.append(y)
                Zn.append(z)

        Xn = np.array(Xn)
        Yn = np.array(Yn)
        Zn = np.array(Zn)

        ZI_new = griddata((Xn, Yn), Zn, (XI, YI), method='linear')

        # -----------------------------
        # 3. Create Plotly surfaces
        # -----------------------------
        import plotly.graph_objects as go
        fig = go.Figure()

        fig.add_trace(
            go.Surface(
                z=ZI_ahn,
                x=XI,
                y=YI,
                opacity=0.8,
                name="Existing AHN"
            )
        )

        fig.add_trace(
            go.Surface(
                z=ZI_new,
                x=XI,
                y=YI,
                opacity=0.6,
                name="New Dike Design"
            )
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (RD)",
                yaxis_title="Y (RD)",
                zaxis_title="Elevation (m)",
                aspectmode="auto"  # automatically scales axes
            ),
            autosize=True
        )

        fig.show()