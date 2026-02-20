from pathlib import Path

import pytest
import geopandas as gpd
from shapely.geometry import shape
import numpy as np

from app.dike_components.ground_model import GroundModel
from app.dike_components.dike_model import DikeModel


def assert_cost_structure(actual, expected, path="root"):
    assert isinstance(actual, type(expected)), (
        f"type mismatch at {path}: "
        f"{type(actual)} != {type(expected)}"
    )

    # -------------------------------
    # Case 1: unit-based cost leaf
    # -------------------------------
    if isinstance(expected, dict) and {
        "value", "unit_cost", "quantity", "unit"
    }.issubset(expected):
        np.testing.assert_allclose(
            actual["value"], expected["value"], rtol=1e-6,
            err_msg=f"value mismatch at {path}.value",
        )
        np.testing.assert_allclose(
            actual["unit_cost"], expected["unit_cost"], rtol=1e-6,
            err_msg=f"unit_cost mismatch at {path}.unit_cost",
        )
        np.testing.assert_allclose(
            actual["quantity"], expected["quantity"], rtol=1e-6,
            err_msg=f"quantity mismatch at {path}.quantity",
        )
        assert actual["unit"] == expected["unit"], (
            f"unit mismatch at {path}.unit"
        )
        return

    # ---------------------------------
    # Case 2: aggregated cost leaf
    # ---------------------------------
    if isinstance(expected, dict) and "base_cost" in expected:
        np.testing.assert_allclose(
            actual["base_cost"], expected["base_cost"], rtol=1e-6,
            err_msg=f"base_cost mismatch at {path}.base_cost",
        )

        # compare remaining scalar metadata strictly
        for key, value in expected.items():
            if key == "base_cost":
                continue
            assert actual[key] == value, (
                f"mismatch at {path}.{key}: "
                f"{actual[key]} != {value}"
            )
        return

    # -------------------------------
    # Case 3: nested dict (group)
    # -------------------------------
    if isinstance(expected, dict):
        for key, expected_child in expected.items():
            assert key in actual, f"missing key '{key}' at {path}"
            assert_cost_structure(
                actual[key],
                expected_child,
                path=f"{path}.{key}",
            ) if key != "description" else None
        return

    # -------------------------------
    # Case 4: scalar totals
    # -------------------------------
    np.testing.assert_allclose(
        actual, expected, rtol=1e-6,
        err_msg=f"scalar mismatch at {path}",
    )
@pytest.fixture(scope="module")
def dike_model():
    path = Path(__file__).parent.joinpath('test_data/test_berm__ontwerp_3d.geojson')
    gdf_ground = gpd.read_file(path)
    model = DikeModel(_3d_ground_polygon=gdf_ground, complexity='makkelijke maatregel')
    model.ground_model.calculate_volume()

    return model

def test_3d_surface_area_positive(dike_model):
    area = dike_model.ground_model.calculate_total_3d_surface_area()["total_3d_area_m2"]
    assert area > 0


def test_compute_cost_structure(dike_model):
    costs = dike_model.compute_cost(nb_houses=10, road_area=10)
    print(costs["Vastgoedkosten"])

    assert "Directe kosten grondwerk" in costs['Bouwkosten']['Directe Bouwkosten']
    assert "totale_BDBK_grondwerk" in costs['Bouwkosten']['Directe Bouwkosten']["Directe kosten grondwerk"]
    assert "Vastgoedkosten" in costs


def test_compute_cost_monotonic_road_area(dike_model):
    cost_small = dike_model.compute_cost(nb_houses=0, road_area=5)
    cost_large = dike_model.compute_cost(nb_houses=0, road_area=20)

    assert cost_large['Bouwkosten']['Directe Bouwkosten']['Directe kosten infrastructuur']['verwijderen_weg']['value'] > cost_small['Bouwkosten']['Directe Bouwkosten']['Directe kosten infrastructuur']['verwijderen_weg']['value']


def test_groundwork_cost_nonzero(dike_model):
    costs = dike_model.compute_cost(nb_houses=0, road_area=0)

    print(costs)
    EXPECTED_COST_DECOMPOSITION = {'Bouwkosten': {'Directe Bouwkosten': {'Directe kosten grondwerk': {'kosten_opruimen': {'value': 820.8411496489607, 'value_incl_BTW': 993.2177910752424, 'unit_cost': 0.28, 'quantity': 2931.5755344605736, 'unit': 'm2', 'description': 'Opruimen terrein en afvoeren naar stort', 'dimension': None}, 'kosten_maaien': {'value': 58.631510689211474, 'value_incl_BTW': 70.94412793394588, 'unit_cost': 0.02, 'quantity': 2931.5755344605736, 'unit': 'm2', 'description': 'Maaien terreinen en afvoeren maaisel naar stort', 'dimension': None}, 'afgraven_toplaag': {'value': 2172.904398249233, 'value_incl_BTW': 2629.214321881572, 'unit_cost': 3.71, 'quantity': 585.6885170483108, 'unit': 'm3', 'description': 'Ontgraven teelaarde en in tijdelijk depot zetten', 'dimension': None}, 'afgraven_oud_materiaal': {'value': 7192.254989353257, 'value_incl_BTW': 8702.62853711744, 'unit_cost': 3.07, 'quantity': 2342.7540681932433, 'unit': 'm3', 'description': 'Ontgraven zand en in tijdelijk depot zetten', 'dimension': None}, 'hergebruik_oud_materiaal': {'value': 6793.986797760405, 'value_incl_BTW': 8220.72402529009, 'unit_cost': 2.9, 'quantity': 2342.7540681932433, 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen van zand (kern)', 'dimension': None}, 'aanvullen_kern': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 14.54, 'quantity': 0, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) zand (kern)', 'dimension': None}, 'profileren_dijkkern': {'value': 2153.3102, 'value_incl_BTW': 2605.505342, 'unit_cost': 0.73, 'quantity': 2949.74, 'unit': 'm2', 'description': 'Profileren dijkprofiel kern', 'dimension': None}, 'aanbrengen_nieuwe_kleilaag': {'value': 38750.569727331036, 'value_incl_BTW': 46888.189370070555, 'unit_cost': 21.06, 'quantity': 1840.0080592274949, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) klei', 'dimension': None}, 'profileren_nieuwe_kleilaag': {'value': 2359.792, 'value_incl_BTW': 2855.3483199999996, 'unit_cost': 0.8, 'quantity': 2949.74, 'unit': 'm2', 'description': 'Profileren dijkprofiel afdeklaag', 'dimension': None}, 'hergebruik_toplaag': {'value': 2301.7558719998615, 'value_incl_BTW': 2785.1246051198323, 'unit_cost': 3.93, 'quantity': 585.6885170483108, 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen toplaag (teelaarde)', 'dimension': None}, 'aanvullen_toplaag': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 16.9, 'quantity': 0, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) teelaarde', 'dimension': None}, 'profileren_nieuwe_toplaag': {'value': 2307.9993761849805, 'value_incl_BTW': 2792.6792451838264, 'unit_cost': 0.77, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Profileren dijkprofiel teelaarde (eindprofiel)', 'dimension': None}, 'inzaaien_nieuwe_toplaag': {'value': 1019.1166076660953, 'value_incl_BTW': 1233.1310952759752, 'unit_cost': 0.34, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Inzaaien dijkprofiel', 'dimension': None}, 'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk', 'value_excl_BTW': 65931.16262888304, 'value_incl_BTW': 79776.7067809485}}, 'Directe kosten constructies': {'directe_bouwkosten': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 0.0, 'quantity': 0.0, 'unit': '', 'description': '', 'dimension': None}, 'totale_BDBK_constructie': {'description': 'Benoemde directe bouwkosten constructie', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}, 'Directe kosten infrastructuur': {'verwijderen_weg': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 15.0, 'quantity': 0, 'unit': 'm2', 'description': 'Opbreken en afvoeren regionale weg (B=4-7m) (incl. stort-/recyclingskosten)', 'dimension': None}, 'aanleggen_weg': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 54.79, 'quantity': 0, 'unit': 'm2', 'description': 'Leveren en aanbrengen regionale weg (B=4-7m) exclusief bebording verlichting bermen en sloten', 'dimension': None}, 'verwijderen_fietspad': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 10.84, 'quantity': 0, 'unit': 'm2', 'description': 'Opbreken en afvoeren fietspad (B<2m) (incl. stort-/recyclingskosten)', 'dimension': None}, 'aanleggen_fietspad': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 38.61, 'quantity': 0, 'unit': 'm2', 'description': 'Leveren en aanbrengen fietspad (B<2m) exclusief bebording en verlichting', 'dimension': None}}}, 'Indirecte Bouwkosten': {'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk', 'value_excl_BTW': 65931.16262888304, 'value_incl_BTW': 79776.7067809485}, 'totale_BDBK_constructie': {'description': 'Benoemde directe bouwkosten constructie', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}, 'totale_BDBK_infrastructuur': {'description': 'Benoemde directe bouwkosten infrastructuur', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}, 'directe_niet_benoemde_bouwkosten_grondwerk': {'code': 'Q-GGMAKNTD', 'surcharge_percentage': 1.0, 'base_cost': 65931.16262888304, 'value': 659.3116262888304, 'description': ''}, 'directe_niet_benoemde_bouwkosten_constructie': {'code': 'Q-GCMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': ''}, 'directe_niet_benoemde_bouwkosten_infrastructuur': {'code': 'Q-GCMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': ''}, 'pm_kosten': {'code': 'Q-EKABKUKMAN', 'surcharge_percentage': 20.0, 'base_cost': 66590.47425517188, 'value': 13318.094851034377, 'description': ''}, 'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 79908.56910620625, 'value': 5753.41697564685, 'description': ''}, 'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 85661.9860818531, 'value': 4368.761290174508, 'description': ''}, 'totale_directe_bouwkosten': {'description': 'Totale directe bouwkosten', 'value_excl_BTW': 66590.47425517188, 'value_incl_BTW': 80574.47384875799}, 'indirecte_bouwkosten': {'description': 'Indirecte bouwkosten', 'value_excl_BTW': 23440.273116855737, 'value_incl_BTW': 28362.73047139544}, 'totale_bouwkosten': {'description': 'Totale bouwkosten', 'value_excl_BTW': 90030.74737202762, 'value_incl_BTW': 108937.20432015343}}}, 'Engineeringkosten': {'engineering_opdrachtgever': {'code': 'Q-ENGOG1', 'surcharge_percentage': 8.0, 'base_cost': 90030.74737202762, 'value': 7202.45978976221, 'description': 'Engineeringskosten opdrachtgever (EPK) - makkelijk'}, 'engineering_opdrachtnemer': {'code': 'Q-ENGON1', 'surcharge_percentage': 5.9, 'base_cost': 90030.74737202762, 'value': 5311.81409494963, 'description': 'Engineeringskosten opdrachtnemer (schets-, voor-, definitief ontwerp, e.d.) - makkelijk'}, 'onderzoekskosten': {'code': 'Q-OND', 'surcharge_percentage': 1.0, 'base_cost': 90030.74737202762, 'value': 900.3074737202762, 'description': 'Onderzoeken (archeologie, explosieven, LNC, e.d.))'}, 'algemene_kosten': {'code': '', 'surcharge_percentage': 7.200000000000001, 'base_cost': 13414.581358432115, 'value': 965.8498578071125, 'description': ''}, 'winst_en_risico': {'code': '', 'surcharge_percentage': 5.099999999999999, 'base_cost': 14380.431216239229, 'value': 733.4019920282004, 'description': ''}, 'direct_engineering_cost': {'description': 'Totale directe engineeringkosten', 'value_excl_BTW': 13414.581358432117, 'value_incl_BTW': 14719.126887852797}, 'indirect_engineering_cost': {'description': 'Totale indirecte engineeringkosten', 'value_excl_BTW': 1699.251849835313, 'value_incl_BTW': 1864.5012411380892}, 'total_engineering_costs': {'description': 'Totale engineeringkosten', 'value_excl_BTW': 15113.83320826743, 'value_incl_BTW': 16583.628128990887}}, 'Overige bijkomende kosten': {'vergunningen_verzekeringen': {'code': 'Q-VERG', 'surcharge_percentage': 3.0, 'base_cost': 90030.74737202762, 'value': 2700.922421160829, 'description': 'Vergunningen, heffingen en verzekeringen'}, 'kabels_leidingen': {'code': 'Q-KL', 'surcharge_percentage': 1.0, 'base_cost': 90030.74737202762, 'value': 900.3074737202762, 'description': 'Kabels & leidingen'}, 'planschade_inpassingsmaatregelen': {'code': 'Q-PLAN', 'surcharge_percentage': 4.0, 'base_cost': 90030.74737202762, 'value': 3601.229894881105, 'description': 'Planschade & inpassingsmaatregelen'}, 'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 7202.45978976221, 'value': 518.5771048628791, 'description': 'Algemene kosten (AK)'}, 'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 7721.036894625089, 'value': 393.7728816258795, 'description': 'Winst & risico'}, 'direct_general_costs': {'description': 'Totale directe algemene kosten', 'value_excl_BTW': 7202.45978976221, 'value_incl_BTW': 8714.976345612275}, 'indirect_general_costs': {'description': 'Totale indirecte algemene kosten', 'value_excl_BTW': 912.3499864887585, 'value_incl_BTW': 1103.9434836513979}, 'total_general_costs': {'description': 'Totale algemene kosten', 'value_excl_BTW': 8114.809776250968, 'value_incl_BTW': 9818.919829263672}}, 'Risicoreservering': {'code': '', 'surcharge_percentage': 10.0, 'base_cost': 113259.39035654602, 'value': 11325.939035654601, 'description': 'Objectoverstijgende risicoreservering (makkelijk)'}, 'Vastgoedkosten': {'direct_benoemd_real_estate_cost': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 700000.0, 'quantity': 0, 'unit': 'panden', 'description': '', 'dimension': None}, 'direct_niet_benoemd_real_estate_cost': {'code': 'Q-GVMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Nader te detailleren directe bouwkosten'}, 'indirect_real_estate_cost': {'code': 'Q-GVMAKIND', 'surcharge_percentage': 7.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Indirecte vastgoedkosten'}, 'real_estate_risk_cost': {'code': 'Q-GVMAKNBO', 'surcharge_percentage': 15.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Niet benoemd objectrisico vastgoed'}, 'total_real_estate_costs': {'description': 'Totale vastgoedkosten', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}}

    # --- Full recursive comparison ---
    for category, expected_items in EXPECTED_COST_DECOMPOSITION.items():
        assert category in costs, f"missing top-level category '{category}'"
        assert_cost_structure(
            costs[category],
            expected_items,
            path=category,
        )
