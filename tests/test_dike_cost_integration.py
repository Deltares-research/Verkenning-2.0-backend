import pytest
import geopandas as gpd
from shapely.geometry import shape
import numpy as np

from app.dike_components.ground_model import GroundModel
from app.dike_components.dike_model import DikeModel


@pytest.fixture(scope="module")
def dike_model():
    gdf_ground = gpd.read_file('tests/test_data/test_berm__ontwerp_3d.geojson')
    model = DikeModel(gdf_ground)
    model.ground_model.calculate_volume()

    return model

def test_3d_surface_area_positive(dike_model):
    area = dike_model.calculate_total_3d_surface_area()["total_3d_area_m2"]
    assert area > 0


def test_compute_cost_structure(dike_model):
    costs = dike_model.compute_cost(nb_houses=10, road_area=10, complexity='makkelijke maatregel')

    assert "Directe kosten grondwerk" in costs
    assert "groundwork_cost" in costs["Directe kosten grondwerk"]
    assert "Vastgoedkosten" in costs
    assert "road_cost" in costs["Vastgoedkosten"]


def test_compute_cost_monotonic_road_area(dike_model):
    cost_small = dike_model.compute_cost(nb_houses=0, road_area=5, complexity='makkelijke maatregel')
    cost_large = dike_model.compute_cost(nb_houses=0, road_area=20, complexity='makkelijke maatregel')

    assert cost_large["Vastgoedkosten"]["road_cost"] > cost_small["Vastgoedkosten"]["road_cost"]


def test_groundwork_cost_nonzero(dike_model):
    costs = dike_model.compute_cost(nb_houses=0, road_area=0, complexity='makkelijke maatregel')
    EXPECTED_COST_DECOMPOSITION = {
        'Directe kosten grondwerk': {'preparation_cost': 914.04, 'groundwork_cost': 76489.88},
        'Directe kosten constructies': {'directe_kosten_constructie': 0.0},
        'Engineeringkosten': {'epk_cost': 8355.92,},
        'Vastgoedkosten': {'house_cost': 0, 'road_cost': 0.0}}
    np.testing.assert_allclose(costs["Directe kosten grondwerk"]["groundwork_cost"], EXPECTED_COST_DECOMPOSITION['Directe kosten grondwerk']['groundwork_cost'], rtol=1e-2)

    #check if the full cost dict matches with floats within tolerance
    for key in EXPECTED_COST_DECOMPOSITION:
        for subkey in EXPECTED_COST_DECOMPOSITION[key]:
            np.testing.assert_allclose(costs[key][subkey], EXPECTED_COST_DECOMPOSITION[key][subkey], rtol=1e-2)