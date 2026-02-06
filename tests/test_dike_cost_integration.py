import pytest
import geopandas as gpd
from shapely.geometry import shape
import numpy as np

from app.dike_components.ground_model import GroundModel
from app.dike_components.dike_model import DikeModel


@pytest.fixture(scope="module")
def dike_model():
    gdf_ground = gpd.read_file('tests/test_data/test_berm__ontwerp_3d.geojson')
    model = DikeModel(_3d_ground_polygon=gdf_ground, complexity='makkelijke maatregel')
    model.ground_model.calculate_volume()

    return model

def test_3d_surface_area_positive(dike_model):
    area = dike_model.ground_model.calculate_total_3d_surface_area()["total_3d_area_m2"]
    assert area > 0


def test_compute_cost_structure(dike_model):
    costs = dike_model.compute_cost(nb_houses=10, road_area=10)

    assert "Directe kosten grondwerk" in costs
    assert "totale_BDBK_grondwerk" in costs["Directe kosten grondwerk"]
    assert "Vastgoedkosten" in costs
    assert "road_cost" in costs["Vastgoedkosten"]


def test_compute_cost_monotonic_road_area(dike_model):
    cost_small = dike_model.compute_cost(nb_houses=0, road_area=5)
    cost_large = dike_model.compute_cost(nb_houses=0, road_area=20)

    assert cost_large["Vastgoedkosten"]["road_cost"] > cost_small["Vastgoedkosten"]["road_cost"]


def test_groundwork_cost_nonzero(dike_model):
    costs = dike_model.compute_cost(nb_houses=0, road_area=0)

    EXPECTED_COST_DECOMPOSITION = {
        "Directe kosten grondwerk": {
            "preparation_cost": {
                "value": 914.0393709229633,
                "unit_cost": 0.30000000000000004,
                "quantity": 3046.7979030765437,
                "unit": "m2",
            },
            "afgraven_grasbekleding_cost": {
                "value": 968.8817331783408,
                "unit_cost": 3.71,
                "quantity": 261.15410597798945,
                "unit": "m3",
            },
            "afgraven_kleilaag_cost": {
                "value": 3206.97242140971,
                "unit_cost": 3.07,
                "quantity": 1044.6164239119578,
                "unit": "m3",
            },
            "herkeuren_kleilaag_cost": {
                "value": 3029.3876293446774,
                "unit_cost": 2.9,
                "quantity": 1044.6164239119578,
                "unit": "m3",
            },
            "aanvullen_kern_cost": {
                "value": 15291.223992375042,
                "unit_cost": 14.54,
                "quantity": 1051.6660242348723,
                "unit": "m3",
            },
            "profieleren_dijkkern_cost": {
                "value": 2188.1033046948514,
                "unit_cost": 0.73,
                "quantity": 2997.4017872532213,
                "unit": "m2",
            },
            "aanbregen_nieuwe_kleilaag_cost": {
                "value": 40786.03803649607,
                "unit_cost": 21.06,
                "quantity": 1936.6589760919314,
                "unit": "m3",
            },
            "profieleren_vannieuwe_kleilaag_cost": {
                "value": 2397.921429802577,
                "unit_cost": 0.8,
                "quantity": 2997.4017872532213,
                "unit": "m3",
            },
            "hergebruik_teelaarde_cost": {
                "value": 1026.3356364934987,
                "unit_cost": 3.93,
                "quantity": 261.15410597798945,
                "unit": "m3",
            },
            "aanvullen_teelaarde_cost": {
                "value": 5392.096125332834,
                "unit_cost": 16.9,
                "quantity": 319.05894232738666,
                "unit": "m3",
            },
            "profieleren_nieuwe_graslaag_cost": {
                "value": 1288.882768518885,
                "unit_cost": 0.43,
                "quantity": 2997.4017872532213,
                "unit": "m2",
            },
            "totale_BDBK_grondwerk": 76489.88244856945,
        },
        "Directe kosten constructies": {
            "totale_BDBK_constructie": 0.0,
        },
        "Engineeringkosten": {
            "epk_cost": 8355.916696941656,
        },
        "Vastgoedkosten": {
            "house_cost": 0,
            "road_cost": 0.0,
            "total_real_estate_costs": 0.0,
        },
    }

    # --- Full comparison ---
    for category, expected_items in EXPECTED_COST_DECOMPOSITION.items():
        assert category in costs

        for key, expected in expected_items.items():
            assert key in costs[category]

            actual = costs[category][key]

            if isinstance(expected, dict):
                np.testing.assert_allclose(actual["value"], expected["value"], rtol=1e-6)
                np.testing.assert_allclose(actual["unit_cost"], expected["unit_cost"], rtol=1e-6)
                np.testing.assert_allclose(actual["quantity"], expected["quantity"], rtol=1e-6)
                assert actual["unit"] == expected["unit"]
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-6)
