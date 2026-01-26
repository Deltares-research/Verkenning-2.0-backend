#make a gpd plot of the above geojson
import geopandas as gpd
import matplotlib.pyplot as plt
import pytest
from shapely.geometry import shape
import json
import numpy as np
from pathlib import Path
from app.dike_components.structure_model import StructureModel
from app.dike_components.onverankerde_damwand_model import OnverankerdeDamwandModel

#define test for reading the geojson with properties, and check if the properties are correct
@pytest.mark.parametrize("filepath, expected_diepte, expected_type", [
    (Path('tests/test_data/test_damwand_input_lines_with_properties.geojson'), -2.0, 'Onverankerde damwand'),
])
def test_read_geojson_with_properties(filepath, expected_diepte, expected_type):
    gdf = gpd.read_file(filepath)
    for _, row in gdf.iterrows():
        assert row['diepte'] == expected_diepte
        assert row['type'] == expected_type


@pytest.mark.parametrize("filepath, expected_cost", [
    (Path('tests/test_data/test_damwand_input_lines_with_properties.geojson'), 3122321.02),
])
def test_compute_direct_cost_of_unanchored_sheet_pile_wall(filepath, expected_cost):
    # Load test geojson data
    filepath = Path('tests/test_data/test_damwand_input_lines_with_properties.geojson')
    gdf = gpd.read_file(filepath)

    # Create OnverankerdeDamwandModel instance
    onverankerd_model = OnverankerdeDamwandModel(gdf)

    # Compute cost
    onverankerd_model.compute_cost()

    total_cost = onverankerd_model.cost_components['directe_bouwkosten']
    # Check if the computed cost is as expected (example expected value)
    expected_cost = 417198.65  # Replace with the actual expected cost for the test data
    np.testing.assert_almost_equal(total_cost, expected_cost, decimal=2)


