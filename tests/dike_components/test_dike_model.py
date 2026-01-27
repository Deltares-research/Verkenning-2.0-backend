from app.dike_components.onverankerde_damwand_model import OnverankerdeDamwandModel
from app.dike_components.dike_model import DikeModel
from app.dike_components.ground_model import GroundModel
import geopandas as gpd

import pytest
import numpy as np




#test if the dike model reads the structure and ground models correctly
@pytest.fixture(scope="module")
def gdf_structure():
    #default output is Onverankerde damwand
    return gpd.read_file('tests/test_data/test_damwand_input_lines_with_properties.geojson')

@pytest.fixture(scope="module")
def gdf_ground():
    return gpd.read_file('tests/test_data/test_berm__ontwerp_3d.geojson')


def test_dike_model_initialization_with_structure_and_ground(gdf_structure, gdf_ground):
    dike_model = DikeModel(_3d_ground_polygon = gdf_ground, _2d_structure = gdf_structure)
    assert hasattr(dike_model, 'ground_model')
    assert hasattr(dike_model, 'structure_model')
    assert dike_model.structure_model.constructietype == 'Onverankerde damwand'

def test_dike_model_initialization_with_structure_only(gdf_structure):
    dike_model = DikeModel(_2d_structure = gdf_structure)
    assert not hasattr(dike_model, 'ground_model')
    assert hasattr(dike_model, 'structure_model')
    assert dike_model.structure_model.constructietype == 'Onverankerde damwand'

def test_dike_model_initialization_with_ground_only(gdf_ground):
    dike_model = DikeModel(_3d_ground_polygon = gdf_ground)
    assert hasattr(dike_model, 'ground_model')
    assert not hasattr(dike_model, 'structure_model')

def test_dike_model_cost_computation_with_ground_only(gdf_ground):
    dike_model = DikeModel(_3d_ground_polygon = gdf_ground, complexity='makkelijke maatregel')
    cost_dict = dike_model.compute_cost(nb_houses=0, road_area=0)

    np.testing.assert_allclose(sum(cost_dict['Directe kosten grondwerk'].values()), 152979.76)
    np.testing.assert_allclose(cost_dict['Directe kosten grondwerk']['totale_BDBK_grondwerk'], 76489.88)
    assert sum(cost_dict['Directe kosten constructies'].values()) == 0.0
    assert sum(cost_dict['Vastgoedkosten'].values()) == 0.0

def test_dike_model_cost_computation_with_structure_only(gdf_structure):
    dike_model = DikeModel(_2d_structure = gdf_structure, complexity='makkelijke maatregel')
    cost_dict = dike_model.compute_cost(nb_houses=0, road_area=0)

    assert sum(cost_dict['Directe kosten grondwerk'].values()) == 0.0
    np.testing.assert_allclose(float(sum(cost_dict['Directe kosten constructies'].values())), 417198.65)
    assert sum(cost_dict['Vastgoedkosten'].values()) == 0.0

def test_dike_model_cost_computation_with_both(gdf_structure, gdf_ground):
    dike_model = DikeModel(_3d_ground_polygon = gdf_ground, _2d_structure = gdf_structure, complexity='makkelijke maatregel')
    cost_dict = dike_model.compute_cost(nb_houses=5, road_area=15)

    np.testing.assert_allclose(float(sum(cost_dict['Directe kosten grondwerk'].values())), 152979.76)
    np.testing.assert_allclose(float(sum(cost_dict['Directe kosten constructies'].values())), 417198.65)
    np.testing.assert_allclose(float(sum(cost_dict['Vastgoedkosten'].values())), 7002093.7)

def test_dike_model_cost_computation_with_none():
    dike_model = DikeModel(complexity='makkelijke maatregel')
    cost_dict = dike_model.compute_cost(nb_houses=0, road_area=0)

    assert sum(cost_dict['Directe kosten grondwerk'].values()) == 0.0
    assert sum(cost_dict['Directe kosten constructies'].values()) == 0.0
    assert sum(cost_dict['Vastgoedkosten'].values()) == 0.0

def test_initialize_dike_model_with_heavescreen(gdf_structure):
    gdf_structure_heavescreen = gdf_structure.copy()
    gdf_structure_heavescreen.loc[0,'type'] = 'Heavescherm'
    dike_model = DikeModel(_2d_structure = gdf_structure_heavescreen)

    #check if structure model is Heavescherm
    assert dike_model.structure_model.constructietype == 'Heavescherm'

def test_initialize_dike_model_with_invalid_type(gdf_structure):
    gdf_structure_invalid = gdf_structure.copy()
    gdf_structure_invalid.loc[0,'type'] = 'Niet geimplementeerde constructie'

    with pytest.raises(ValueError, match="Onbekend constructietype: Niet geimplementeerde constructie"):
        dike_model = DikeModel(_2d_structure = gdf_structure_invalid)

def test_dike_model_cost_computation_with_heavescreen(gdf_structure):
    gdf_structure_heavescreen = gdf_structure.copy()
    gdf_structure_heavescreen.loc[0,'type'] = 'Heavescherm'
    dike_model = DikeModel(_2d_structure = gdf_structure_heavescreen, complexity='makkelijke maatregel')
    cost_dict = dike_model.compute_cost(nb_houses=0, road_area=0)

    np.testing.assert_allclose(float(sum(cost_dict['Directe kosten constructies'].values())), 375517.53)  #example value