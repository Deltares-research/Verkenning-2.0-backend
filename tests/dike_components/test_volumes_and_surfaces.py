from pathlib import Path

from app.dike_components.dike_model import DikeModel
import geopandas as gpd

import pytest
import numpy as np



@pytest.fixture(scope="module")
def gdf_ground():
    path = Path(__file__).parent.parent.joinpath('test_data/test_berm__ontwerp_3d.geojson')
    return gpd.read_file(path)

@pytest.fixture(scope="module")
def gdf_ground_cut_and_fill():
    path = Path(__file__).parent.parent.joinpath('test_data/test_berm__ontwerp_3d_afgraven.geojson')
    return gpd.read_file(path)



def test_comparison_area_flat_surface(gdf_ground):
    """
    This test compares for a planar surface (the design surface in this case) the 3D area from two different methods:
        - Newells (analytical, so most accurate) only valid for planar surfaces
        - TIN (triangulation) valid for both planar and non-planar surfaces
    """
    dike_model = DikeModel(_3d_ground_polygon = gdf_ground)
    ground_model = dike_model.ground_model


    newells_method = ground_model.calculate_total_3d_surface_area()['total_3d_area_m2']
    TIN_method = ground_model.calculate_3d_surface_TIN(height_source='design_edges')['area'] # trully planar input
    TIN_method_grid = ground_model.calculate_3d_surface_TIN(height_source='design')['area']  # discretized input

    expected_newells = 2997.4017872532213
    expected_tin = 3002.383799692105
    expected_tin_grid = 2953.0820040953095  # This has approximation errors. Ideally closer to the other 2.

    assert newells_method > 0
    assert TIN_method > 0
    assert np.isclose(TIN_method, newells_method, rtol=0.005)  # allow 0.5% tolerance. actual difference is 0.166% here
    assert newells_method == pytest.approx(expected_newells, rel=1e-9)
    assert TIN_method == pytest.approx(expected_tin, rel=1e-9)

def test_area_ahn_surface(gdf_ground):
    dike_model = DikeModel(_3d_ground_polygon = gdf_ground)
    ground_model = dike_model.ground_model

    TIN_method = ground_model.calculate_3d_surface_TIN(height_source='ahn')['area']

    expected_tin = 2931.5755344605736
    assert TIN_method == pytest.approx(expected_tin, rel=1e-3)

def test_ahn_full_and_envelop(gdf_ground_cut_and_fill):
    dike_model = DikeModel(_3d_ground_polygon = gdf_ground_cut_and_fill)
    ground_model = dike_model.ground_model

    full_AHN_surface = ground_model.calculate_3d_surface_TIN(height_source='ahn')['area']
    envelop_AHN_surface = ground_model.calculate_3d_surface_TIN(height_source='ahn', exclude_points_where_ahn_above_design=True)['area']

    EXPECTED_FULL_AHN_SURFACE = 2428.820012439969
    EXPECTED_ENVELOP_AHN_SURFACE = 1844.8961842200986

    assert full_AHN_surface == pytest.approx(EXPECTED_FULL_AHN_SURFACE, rel=1e-9)
    assert envelop_AHN_surface == pytest.approx(EXPECTED_ENVELOP_AHN_SURFACE, rel=1e-9)
    assert full_AHN_surface > envelop_AHN_surface

def test_2d_3d_ruimtebeslag(gdf_ground_cut_and_fill):
    dike_model = DikeModel(_3d_ground_polygon = gdf_ground_cut_and_fill)
    ground_model = dike_model.ground_model

    footprint_2d_area = ground_model.calculate_volume()['area'] # just the 2D footprint of the design
    ruimtebelsag_2d_area = ground_model.calculate_ruimtebeslag_2d()['total_area_m2']  # 2D area of the design surface where design > AHN

    newells_method = ground_model.calculate_total_3d_surface_area()['total_3d_area_m2'] # 3D area of the design surface with Newell analytical method
    TIN_method = ground_model.calculate_3d_surface_TIN(height_source='design_edges')['area']  # 3D area of the design surface with TIN method

    ruimtebelsag_3d_area = ground_model.calculate_3d_surface_TIN(height_source='design', exclude_points_where_ahn_above_design=True)['area']  # 3D area of the design surface where design > AHN


    assert footprint_2d_area > ruimtebelsag_2d_area
    assert newells_method > footprint_2d_area
    assert ruimtebelsag_3d_area > ruimtebelsag_2d_area
    assert ruimtebelsag_3d_area < newells_method


def test_V3_V4_V5(gdf_ground):
    dike_model = DikeModel(_3d_ground_polygon = gdf_ground)
    ground_model = dike_model.ground_model
    volumes = ground_model.calculate_volume_v3_v4_v5()
    print(volumes)
    #allclose for volumes rtol = 1e-2
    assert volumes[0] == pytest.approx(580.2130483053766, rel=1e-2)  # V3
    assert volumes[1] == pytest.approx(1936.658976091931, rel=1e-2)  # V4
    assert volumes[2] == pytest.approx(790.5119182568828, rel=1e-2)  # V5







    






