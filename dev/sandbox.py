from pathlib import Path

from pyproj import Transformer
from shapely.geometry.linestring import LineString

from app.AHN5 import API_ahn
from app.AHN_raster_API import AHN4, AHN4_API
from app.volume_calc import DikeModel
import time
import geopandas as gpd
xmin=182926.38
xmax=183419.62
ymin=430891.89
ymax=431154.24


# api = API_ahn()
# point = (xmin, ymin)
# api.collect_ahn_data(point)
# api.get_ahn_list([(xmin, ymin), (xmax, ymax)])


# print(api.AHN_data)

# model = DikeModel()
# model.calculate_volume()


linestring_test = Path(r'C:\Users\hauth\repositories\Verkenning-2.0\backend\test_data\test_line_dike.shp')
design_export_3d = gpd.read_file(Path(r'C:\Users\hauth\repositories\Verkenning-2.0\backend\test_data\ontwerp_export_3d(1).geojson'))
print(design_export_3d)

gdf = gpd.read_file(linestring_test)
ls_wgs = gdf.geometry.iloc[0]
transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)  # Convert in RD coordinates
linestring = LineString([transformer.transform(x, y) for x, y in ls_wgs.coords])


test_api = AHN4()
t1 = time.time()
# l1, z1 = test_api.get_elevation_from_line(linestring, raster=None)
t2 = time.time()
print(f"Time taken: {t2 - t1} seconds")


#
# test_api2 =  AHN4Optimized()
# t1 = time.time()
# l2, z2 = test_api2.get_elevation_from_line(linestring, raster=None, spacing=0.5)
# t2 = time.time()
#
# print(f"Time taken optimized: {t2 - t1} seconds")
#
# a=2
# # print(l1)
# print(l2)
#
# # print(z1)
# print(z2)
#
# # plot matplotlib graph
# import matplotlib.pyplot as plt
# # plt.plot(l1, z1, label='AHN4')
# plt.plot(l2, z2, label='AHN4 Optimized')
# plt.xlabel('Distance (m)')
# plt.ylabel('Elevation (m)')
# plt.title('Elevation Profile Comparison')
# plt.legend()
# plt.show()


d = DikeModel(design_export_3d)
d.calculate_volume_matthias()
d.plot_existing_and_new_surface_plotly()
t3 = time.time()
print(f"Total volume calculation time: {t3 - t2} seconds")

# Test 1 VertiGIS, 13745 fill and 0 cut
# Test 1 15044.947265625 Total cut (m3): 1169.4258813858032
