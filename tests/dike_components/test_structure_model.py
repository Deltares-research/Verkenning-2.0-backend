from app.dike_components.onverankerde_damwand_model import OnverankerdeDamwandModel
from app.dike_components.dike_model import DikeModel
import geopandas as gpd

import pytest
import numpy as np


@pytest.fixture(scope="module")
def gdf_structure():
    #default output is Onverankerde damwand
    return gpd.read_file('tests/test_data/test_damwand_input_lines_with_properties.geojson')


#test for checking if gdf is read correctly
@pytest.mark.parametrize("expected_diepte, expected_type, expected_wandlengte", [(-2.0, 'Onverankerde damwand', 9.346578386235745)])
def test_structure_model_properties(expected_diepte, expected_type,expected_wandlengte,gdf_structure):
    onverankerde_damwand = OnverankerdeDamwandModel(location=gdf_structure)

    assert onverankerde_damwand.diepte == expected_diepte
    assert onverankerde_damwand.constructietype == expected_type
    assert onverankerde_damwand.wandlengte == expected_wandlengte

def test_get_screen_length(gdf_structure):
    depths = [-1.5, -2.0, -3.0, -4.0]
    expected_lengths = [8.85, 9.35, 10.35, 11.35]
    outcomes = []
    for depth, expected_length in zip(depths, expected_lengths):
        #copy object to avoid modifying original
        new_object = OnverankerdeDamwandModel(location=gdf_structure)
        new_object.diepte = depth
        new_object.get_screen_length()   
        np.testing.assert_allclose(new_object.wandlengte, expected_length, rtol=1e-2)

def test_get_length_from_geometry(gdf_structure):
    _expected_length = 187.21  # Expected length based on the test data
    structure_object = OnverankerdeDamwandModel(location=gdf_structure)

    structure_object.get_length_from_geometry()
    np.testing.assert_allclose(structure_object.length, _expected_length, rtol=1e-2)
# determine_length_from_depth
