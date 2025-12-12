"""
Dike volume calculation using AHN4 elevation data.
"""
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

from AHN_raster_API import AHN4_API

xmin = 182926.38
xmax = 183419.62
ymin = 430891.89
ymax = 431154.24
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
    return Polygon(exterior_coords_3d)



# from arcgis.geometry import Point, Polygon, Multipoint, project
# from arcgis.gis import GIS

class DikeModel:
    def __init__(self, design_export_3d: gpd.GeoDataFrame):
        self.grid_size = 1
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

    def compute_volumes(self, current_elev, new_elev, cell_area=1.0):
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

    def calculate_volume_matthias(self):
        """
        Compute fill and excavation volumes for all polygons using a single global grid
        and single AHN raster query.
        
        ✅ FIXED: Now interpolates design heights at each grid point instead of using mean!
        """

        # 1) Combine polygons to get full extent
        combined_poly = unary_union(self.design_export_3d.geometry)
        
        print("\n=== DESIGN POLYGONS ===")
        for idx, row in self.design_export_3d.iterrows():
            poly = row.geometry
            print(f"\nPolygon {idx} - {row.get('name', 'unnamed')}:")
            print(f"  Bounds: {poly.bounds}")
            z_values = [coord[2] for coord in poly.exterior.coords]
            print(f"  Z-values: Min={min(z_values):.2f}m, Max={max(z_values):.2f}m, Mean={np.mean(z_values):.2f}m")
            print(f"  ⚠️  This is a SLOPED surface - need to interpolate, not use mean!")

        # 2) Generate a global grid
        time1 = time.time()
        grid_pts_global = self.polygon_grid_2d_vectorized(combined_poly, cellsize=self.grid_size)
        time2 = time.time()
        print(f"\n=== GRID GENERATION ===")
        print(f"Grid: {len(grid_pts_global)} points, size={self.grid_size}m, time={time2-time1:.3f}s")

        # 3) Get the AHN elevations for the grid
        time3 = time.time()
        elev_global = self.get_elevations(AHN4_API(resolution=self.grid_size), combined_poly, grid_pts_global)
        time4 = time.time()
        
        nan_count = np.isnan(elev_global).sum()
        valid_count = len(elev_global) - nan_count
        
        print(f"\n=== AHN ELEVATIONS ===")
        print(f"Valid: {valid_count}/{len(elev_global)} ({100*valid_count/len(elev_global):.1f}%)")
        
        if valid_count == 0:
            print("⚠️  ERROR: NO VALID ELEVATION DATA!")
            return {'fill_volume': 0.0, 'cut_volume': 0.0, 'total_volume': 0.0, 'area': 0.0, 'grid_points': 0}

        # 4) Precompute masks for each polygon
        masks = []
        print("\n=== POLYGON MASKS ===")
        for idx, row in self.design_export_3d.iterrows():
            poly = row.geometry
            path = Path(np.array([[x, y] for x, y, *_ in poly.exterior.coords]))
            mask = path.contains_points(grid_pts_global)
            points_in_poly = np.sum(mask)
            print(f"Polygon {idx} - {row.get('name', 'unnamed')}: {points_in_poly} grid points")
            masks.append(mask)

        # 5) Compute volumes per polygon WITH INTERPOLATION
        tot_volume_fill, tot_volume_cut = 0.0, 0.0

        print("\n=== VOLUME CALCULATIONS (INTERPOLATED) ===")
        for idx, row in self.design_export_3d.iterrows():
            poly = row.geometry
            mask = masks[idx]
            
            # Get grid points inside this polygon
            grid_pts_in_poly = grid_pts_global[mask]
            elev_poly = elev_global[mask]
            
            if len(grid_pts_in_poly) == 0:
                print(f"\nPolygon {idx}: No grid points, skipping")
                continue
            
            # ✅ FIX: Interpolate design heights at each grid point
            poly_coords_3d = np.array(poly.exterior.coords)
            
            print(f"\nPolygon {idx} - {row.get('name', 'unnamed')}:")
            print(f"  Interpolating design heights for {len(grid_pts_in_poly)} grid points...")
            
            # Use scipy griddata to interpolate Z at each grid point
            design_heights = griddata(
                points=poly_coords_3d[:, :2],        # XY coordinates of polygon vertices
                values=poly_coords_3d[:, 2],         # Z values at vertices
                xi=grid_pts_in_poly,                 # Grid points where to evaluate
                method='linear',                      # Linear interpolation
                fill_value=np.mean(poly_coords_3d[:, 2])  # Fallback for points outside convex hull
            )
            
            print(f"  Design heights (interpolated): Min={np.nanmin(design_heights):.2f}m, Max={np.nanmax(design_heights):.2f}m")
            
            # Filter valid AHN elevations
            valid_mask = ~np.isnan(elev_poly)
            design_heights_valid = design_heights[valid_mask]
            elev_poly_valid = elev_poly[valid_mask]
            
            print(f"  AHN ground: Min={np.nanmin(elev_poly_valid):.2f}m, Max={np.nanmax(elev_poly_valid):.2f}m")
            print(f"  Valid points for volume calc: {len(elev_poly_valid)}")
            
            # Compute volume with interpolated heights
            dV = design_heights_valid - elev_poly_valid
            fill = np.sum(dV[dV > 0] * self.grid_size ** 2)
            cut = np.sum(-dV[dV < 0] * self.grid_size ** 2)
            
            print(f"  Results: fill={fill:.2f} m³, cut={cut:.2f} m³")

            tot_volume_fill += fill
            tot_volume_cut += cut

        print(f"\n=== FINAL TOTALS ===")
        print(f"Total fill (m³): {tot_volume_fill:.2f}")
        print(f"Total cut (m³): {tot_volume_cut:.2f}")
        print(f"Net difference (m³): {tot_volume_fill - tot_volume_cut:.2f}")
        
        return {
            'fill_volume': tot_volume_fill,
            'cut_volume': tot_volume_cut,
            'total_volume': tot_volume_fill - tot_volume_cut,
            'area': len(grid_pts_global) * (self.grid_size ** 2),
            'grid_points': len(grid_pts_global)
        }

    def compute_volumes(self, current_elev, new_elev, cell_area=1.0):
        """
        OLD METHOD - kept for compatibility
        """
        dV = new_elev - current_elev
        fill = np.sum(dV[dV > 0] * cell_area)
        cut = np.sum(-dV[dV < 0] * cell_area)
        return fill, cut

    def plot_existing_and_new_surface(
            self,
            title="Existing vs New Dike Surface"
    ):
        """
        Plot both the existing (AHN) elevation surface and the new 3D design surface in matplotlib

        Parameters
        ----------
        grid_points : Nx2 array
            Grid points used for AHN sampling.
        current_elev : array
            Elevations from AHN corresponding to grid_points.
        design_polygons : list(Polygon)
            List of 3D shapely Polygons representing the new dike geometry.
        title : str
            Plot title.
        """

        # ----------------------------------------
        # 1. Prepare AHN grid (irregular -> regular grid)
        # ----------------------------------------

        grid_points = self.grid_2d
        current_elev = self.elevation
        design_polygons = self.design_export_3d["geometry"].tolist()

        X = grid_points[:, 0]
        Y = grid_points[:, 1]
        Z = current_elev

        xi = np.linspace(X.min(), X.max(), 200)
        yi = np.linspace(Y.min(), Y.max(), 200)
        XI, YI = np.meshgrid(xi, yi)

        ZI_ahn = griddata((X, Y), Z, (XI, YI), method='linear')

        # ----------------------------------------
        # 2. Extract all vertices of new 3D polygons
        # ----------------------------------------
        Xn, Yn, Zn = [], [], []

        for poly in design_polygons:
            for x, y, z in poly.exterior.coords:
                Xn.append(x)
                Yn.append(y)
                Zn.append(z)

        Xn = np.array(Xn)
        Yn = np.array(Yn)
        Zn = np.array(Zn)

        # Interpolate new surface onto same grid for comparison
        ZI_new = griddata((Xn, Yn), Zn, (XI, YI), method='linear')

        # ----------------------------------------
        # 3. Plot both surfaces
        # ----------------------------------------
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        surf1 = ax.plot_surface(
            XI, YI, ZI_ahn,
            # cmap="terrain",
            alpha=0.7,
            linewidth=0
        )

        surf2 = ax.plot_surface(
            XI, YI, ZI_new,
            # cmap="viridis",
            alpha=0.5,
            linewidth=0
        )

        ax.set_title(title)
        ax.set_xlabel("X (RD)")
        ax.set_ylabel("Y (RD)")
        ax.set_zlabel("Elevation (m)")

        plt.show()

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

    def calculate_volume(model):
        import time
        t1 = time.time()
        grid_size = model.gridSize

        # Elevation sampler equivalent: You must replace with your own function: GET AHN5
        # elevation_sampler = model.elevationSampler

        # Original extent
        # extent = model["meshGraphic"]["geometry"]["extent"]

        # RD New spatial reference
        rd_new_sr = {"wkid": 28992}

        # Project extent polygon to RD New
        # extent_polygon = ({
        #     "rings": [[
        #         [xmin, ymin],
        #         [xmax, ymin],
        #         [xmax, ymax],
        #         [xmin, ymax],
        #         [xmin, ymin]
        #     ]],
        #     "spatialReference": extent["spatialReference"]  #TODO replace
        # })

        # projected_extent = project(extent_polygon, rd_new_sr)
        # rd_extent = projected_extent.extent

        point_coords_for_volume = []
        ground_points = []

        point_count = 0
        valid_points = 0

        rd_extent = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        }

        # Generate RD grid
        x = rd_extent["xmin"]
        query_list = []
        while x <= rd_extent["xmax"]:
            y = rd_extent["ymin"]
            while y <= rd_extent["ymax"]:
                point_count += 1

                rd_point = {"x": x, "y": y, "spatialReference": rd_new_sr}
                # wm_point = project(rd_point, {"wkid": 3857})  # Web Mercator

                # Elevation sampling function required here
                # elevation = model.API_AHN.collect_ahn_data((rd_point["x"], rd_point["y"]))
                elevation = 0
                query_list.append((rd_point["x"], rd_point["y"]))

                if elevation is not None:
                    valid_points += 1
                    point_coords_for_volume.append([rd_point["x"], rd_point["y"], elevation])
                    ground_points.append([rd_point["x"], rd_point["y"]])

                y += grid_size
            x += grid_size

        model.API_AHN.get_ahn_list(query_list)
        print(f"Total grid points: {point_count}")
        print(f"Valid elevation points: {valid_points}")
        t2 = time.time()
        print(f"Grid generation time: {t2 - t1} seconds")

        # # Distance verification
        # if valid_points >= 2:
        #     p1 = (point_coords_for_volume[0][:2], {"wkid": 3857})
        #     p2 = (point_coords_for_volume[1][:2], {"wkid": 3857})
        #
        #     p1_rd = project(p1, rd_new_sr)
        #     p2_rd = project(p2, rd_new_sr)
        #
        #     dx = p2_rd["x"] - p1_rd["x"]
        #     dy = p2_rd["y"] - p1_rd["y"]
        #     dist = math.sqrt(dx * dx + dy * dy)
        #     print(f"Verification dist: {dist}m (expected {grid_size})")
        #
        # if not point_coords_for_volume:
        #     print("No points processed.")
        #     return

        # Query ground elevations (replace with your own DEM sampler)
        ground_elevations = model["groundElevationSampler"](ground_points)

        total_volume = 0
        excavation_volume = 0
        fill_volume = 0

        for i, (x, y, z_ground) in enumerate(ground_elevations):
            z_mesh = point_coords_for_volume[i][2]
            dV = (z_mesh - z_ground) * grid_size * grid_size

            if dV > 0:
                fill_volume += dV
            else:
                excavation_volume += abs(dV)

            total_volume += dV

        # Alpha shape calculation placeholder
        try:
            above_ground_pts = [
                (x, y)
                for (x, y, z_ground), (_, _, z_mesh)
                in zip(ground_elevations, point_coords_for_volume)
                if z_mesh > z_ground
            ]

            # Compute alpha shape using shapely
            from shapely.geometry import MultiPoint
            from shapely.ops import triangulate

            mp = MultiPoint(above_ground_pts)

            # VERY simplified alpha shape placeholder
            alpha_shape = mp.convex_hull
            model["ruimtebeslagGeometry"] = alpha_shape

        except Exception as e:
            print("Alpha shape error:", e)

        # Store results
        model["excavationVolume"] = round(excavation_volume, 2)
        model["fillVolume"] = round(fill_volume, 2)
        model["totalVolumeDifference"] = round(total_volume, 2)

        print("Total volume:", total_volume)
        print("Cut volume:", excavation_volume)
        print("Fill volume:", fill_volume)
