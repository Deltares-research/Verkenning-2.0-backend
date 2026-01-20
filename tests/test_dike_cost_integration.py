import pytest
import geopandas as gpd
from shapely.geometry import shape

from app.dike_model import DikeModel


@pytest.fixture(scope="module")
def dike_model():
    geojson_input = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [5.568245382650625, 51.890881614488016, 8.8],
                        [5.571640932264274, 51.891476523815435, 8.8],
                        [5.57160887667709, 51.89154657512496, 8.9],
                        [5.568213322196268, 51.890951664878685, 8.9],
                        [5.568245382650625, 51.890881614488016, 8.8]
                    ]]
                },
                "properties": {"name": "-34.9m_-26.8m"}
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [5.568213322196268, 51.890951664878685, 8.9],
                        [5.57160887667709, 51.89154657512496, 8.9],
                        [5.571578008239614, 51.891614031932704, 10.1],
                        [5.56818244907187, 51.891019120801595, 10.1],
                        [5.568213322196268, 51.890951664878685, 8.9]
                    ]]
                },
                "properties": {"name": "-26.8m_-19m"}
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [5.56818244907187, 51.891019120801595, 10.1],
                        [5.571578008239614, 51.891614031932704, 10.1],
                        [5.571537245930476, 51.89170310949893, 14],
                        [5.568141680573571, 51.891108197199316, 14],
                        [5.56818244907187, 51.891019120801595, 10.1]
                    ]]
                },
                "properties": {"name": "-19m_-8.7m"}
            }
        ]
    }

    features = []
    for feature in geojson_input["features"]:
        geom = shape(feature["geometry"])
        features.append({"geometry": geom, **feature["properties"]})

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    model = DikeModel(gdf)
    model.calculate_volume()

    return model


def test_3d_surface_area_positive(dike_model):
    area = dike_model.calculate_total_3d_surface_area()["total_3d_area_m2"]
    assert area > 0


def test_compute_cost_structure(dike_model):
    costs = dike_model.compute_cost(nb_houses=10, road_area=10, complexity='makkelijke maatregel')

    assert "Directe bouwkosten" in costs
    assert "Grondwerk" in costs["Directe bouwkosten"]
    assert "Vastgoedkosten" in costs
    assert "Wegen" in costs["Vastgoedkosten"]


def test_compute_cost_monotonic_road_area(dike_model):
    cost_small = dike_model.compute_cost(nb_houses=0, road_area=5, complexity='makkelijke maatregel')
    cost_large = dike_model.compute_cost(nb_houses=0, road_area=20, complexity='makkelijke maatregel')

    assert cost_large["Vastgoedkosten"]["Wegen"] > cost_small["Vastgoedkosten"]["Wegen"]


def test_groundwork_cost_nonzero(dike_model):
    costs = dike_model.compute_cost(nb_houses=0, road_area=0, complexity='makkelijke maatregel')
    EXPECTED_COST_DECOMPOSITION = {
        'Directe bouwkosten': {'Voorbereiding': 0, 'Grondwerk': 284887.0606946243, 'Constructie': 0},
        'Engineeringkosten': {'engineering_cost_EPK': 22790.964855569946,
                              'engineering_cost_schets': 16808.336580982836},
        'Vastgoedkosten': {'Panden': 0, 'Wegen': 0.0}}

    assert costs["Directe bouwkosten"]["Grondwerk"] > 0
    assert costs == EXPECTED_COST_DECOMPOSITION
