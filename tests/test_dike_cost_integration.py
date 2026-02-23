from pathlib import Path

import pytest
import geopandas as gpd
import numpy as np

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
    EXPECTED_COST_DECOMPOSITION = {'Bouwkosten': {'Directe Bouwkosten': {'Directe kosten grondwerk': {'kosten_opruimen': {'value': 820.8537722873807, 'value_incl_BTW': 993.2330644677306, 'unit_cost': 0.28, 'quantity': 2931.620615312074, 'unit': 'm2', 'description': 'Opruimen terrein en afvoeren naar stort', 'dimension': None}, 'kosten_maaien': {'value': 58.63241230624148, 'value_incl_BTW': 70.94521889055218, 'unit_cost': 0.02, 'quantity': 2931.620615312074, 'unit': 'm2', 'description': 'Maaien terreinen en afvoeren maaisel naar stort', 'dimension': None}, 'afgraven_toplaag': {'value': 2172.935046083261, 'value_incl_BTW': 2629.251405760746, 'unit_cost': 3.71, 'quantity': 585.6967779200165, 'unit': 'm3', 'description': 'Ontgraven teelaarde en in tijdelijk depot zetten', 'dimension': None}, 'afgraven_oud_materiaal': {'value': 7192.356432857802, 'value_incl_BTW': 8702.75128375794, 'unit_cost': 3.07, 'quantity': 2342.787111680066, 'unit': 'm3', 'description': 'Ontgraven zand en in tijdelijk depot zetten', 'dimension': None}, 'hergebruik_oud_materiaal': {'value': 6794.082623872191, 'value_incl_BTW': 8220.839974885352, 'unit_cost': 2.9, 'quantity': 2342.787111680066, 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen van zand (kern)', 'dimension': None}, 'aanvullen_kern': {'value': 20008.81486793878, 'value_incl_BTW': 24210.665990205925, 'unit_cost': 14.54, 'quantity': 1376.1220679462713, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) zand (kern)', 'dimension': None}, 'profileren_dijkkern': {'value': 2153.506984185311, 'value_incl_BTW': 2605.7434508642264, 'unit_cost': 0.73, 'quantity': 2950.0095673771384, 'unit': 'm2', 'description': 'Profileren dijkprofiel kern', 'dimension': None}, 'aanbrengen_nieuwe_kleilaag': {'value': 40793.81767998092, 'value_incl_BTW': 49360.519392776914, 'unit_cost': 21.06, 'quantity': 1937.0283798661408, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) klei', 'dimension': None}, 'profileren_nieuwe_kleilaag': {'value': 2360.007653901711, 'value_incl_BTW': 2855.60926122107, 'unit_cost': 0.8, 'quantity': 2950.0095673771384, 'unit': 'm2', 'description': 'Profileren dijkprofiel afdeklaag', 'dimension': None}, 'hergebruik_toplaag': {'value': 2301.788337225665, 'value_incl_BTW': 2785.1638880430546, 'unit_cost': 3.93, 'quantity': 585.6967779200165, 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen toplaag (teelaarde)', 'dimension': None}, 'aanvullen_toplaag': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 16.9, 'quantity': 0, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) teelaarde', 'dimension': None}, 'profileren_nieuwe_toplaag': {'value': 2307.9993761849805, 'value_incl_BTW': 2792.6792451838264, 'unit_cost': 0.77, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Profileren dijkprofiel teelaarde (eindprofiel)', 'dimension': None}, 'inzaaien_nieuwe_toplaag': {'value': 1019.1166076660953, 'value_incl_BTW': 1233.1310952759752, 'unit_cost': 0.34, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Inzaaien dijkprofiel', 'dimension': None}, 'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk', 'value_excl_BTW': 87983.91179449034, 'value_incl_BTW': 106460.5332713333}}, 'Directe kosten constructies': {'directe_bouwkosten': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 0.0, 'quantity': 0.0, 'unit': '', 'description': '', 'dimension': None}, 'totale_BDBK_constructie': {'description': 'Benoemde directe bouwkosten constructie', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}, 'Directe kosten infrastructuur': {'verwijderen_weg': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 15.0, 'quantity': 0, 'unit': 'm2', 'description': 'Opbreken en afvoeren regionale weg (B=4-7m) (incl. stort-/recyclingskosten)', 'dimension': None}, 'aanleggen_weg': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 54.79, 'quantity': 0, 'unit': 'm2', 'description': 'Leveren en aanbrengen regionale weg (B=4-7m) exclusief bebording verlichting bermen en sloten', 'dimension': None}, 'verwijderen_fietspad': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 10.84, 'quantity': 0, 'unit': 'm2', 'description': 'Opbreken en afvoeren fietspad (B<2m) (incl. stort-/recyclingskosten)', 'dimension': None}, 'aanleggen_fietspad': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 38.61, 'quantity': 0, 'unit': 'm2', 'description': 'Leveren en aanbrengen fietspad (B<2m) exclusief bebording en verlichting', 'dimension': None}}}, 'Indirecte Bouwkosten': {'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk', 'value_excl_BTW': 87983.91179449034, 'value_incl_BTW': 106460.5332713333}, 'totale_BDBK_constructie': {'description': 'Benoemde directe bouwkosten constructie', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}, 'totale_BDBK_infrastructuur': {'description': 'Benoemde directe bouwkosten infrastructuur', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}, 'directe_niet_benoemde_bouwkosten_grondwerk': {'code': 'Q-GGMAKNTD', 'surcharge_percentage': 1.0, 'base_cost': 87983.91179449034, 'value': 879.8391179449034, 'description': ''}, 'directe_niet_benoemde_bouwkosten_constructie': {'code': 'Q-GCMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': ''}, 'directe_niet_benoemde_bouwkosten_infrastructuur': {'code': 'Q-GCMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': ''}, 'pm_kosten': {'code': 'Q-EKABKUKMAN', 'surcharge_percentage': 20.0, 'base_cost': 88863.75091243525, 'value': 17772.75018248705, 'description': ''}, 'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 106636.5010949223, 'value': 7677.828078834406, 'description': ''}, 'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 114314.32917375672, 'value': 5830.030787861593, 'description': ''}, 'totale_directe_bouwkosten': {'description': 'Totale directe bouwkosten', 'value_excl_BTW': 88863.75091243525, 'value_incl_BTW': 107525.13860404663}, 'indirecte_bouwkosten': {'description': 'Indirecte bouwkosten', 'value_excl_BTW': 31280.609049183047, 'value_incl_BTW': 37849.53694951149}, 'totale_bouwkosten': {'description': 'Totale bouwkosten', 'value_excl_BTW': 120144.35996161829, 'value_incl_BTW': 145374.6755535581}}}, 'Engineeringkosten': {'engineering_opdrachtgever': {'code': 'Q-ENGOG1', 'surcharge_percentage': 8.0, 'base_cost': 120144.35996161829, 'value': 9611.548796929463, 'description': 'Engineeringskosten opdrachtgever (EPK) - makkelijk'}, 'engineering_opdrachtnemer': {'code': 'Q-ENGON1', 'surcharge_percentage': 5.9, 'base_cost': 120144.35996161829, 'value': 7088.517237735479, 'description': 'Engineeringskosten opdrachtnemer (schets-, voor-, definitief ontwerp, e.d.) - makkelijk'}, 'onderzoekskosten': {'code': 'Q-OND', 'surcharge_percentage': 1.0, 'base_cost': 120144.35996161829, 'value': 1201.4435996161828, 'description': 'Onderzoeken (archeologie, explosieven, LNC, e.d.))'}, 'algemene_kosten': {'code': '', 'surcharge_percentage': 7.200000000000001, 'base_cost': 17901.509634281123, 'value': 1288.908693668241, 'description': ''}, 'winst_en_risico': {'code': '', 'surcharge_percentage': 5.1, 'base_cost': 19190.418327949366, 'value': 978.7113347254176, 'description': ''}, 'direct_engineering_cost': {'description': 'Totale directe engineeringkosten', 'value_excl_BTW': 17901.509634281123, 'value_incl_BTW': 19642.401410124974}, 'indirect_engineering_cost': {'description': 'Totale indirecte engineeringkosten', 'value_excl_BTW': 2267.6200283936587, 'value_incl_BTW': 2488.1422714233504}, 'total_engineering_costs': {'description': 'Totale engineeringkosten', 'value_excl_BTW': 20169.129662674783, 'value_incl_BTW': 22130.543681548326}}, 'Overige bijkomende kosten': {'vergunningen_verzekeringen': {'code': 'Q-VERG', 'surcharge_percentage': 3.0, 'base_cost': 120144.35996161829, 'value': 3604.3307988485485, 'description': 'Vergunningen, heffingen en verzekeringen'}, 'kabels_leidingen': {'code': 'Q-KL', 'surcharge_percentage': 1.0, 'base_cost': 120144.35996161829, 'value': 1201.4435996161828, 'description': 'Kabels & leidingen'}, 'planschade_inpassingsmaatregelen': {'code': 'Q-PLAN', 'surcharge_percentage': 4.0, 'base_cost': 120144.35996161829, 'value': 4805.774398464731, 'description': 'Planschade & inpassingsmaatregelen'}, 'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 9611.548796929463, 'value': 692.0315133789213, 'description': 'Algemene kosten (AK)'}, 'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 10303.580310308384, 'value': 525.4825958257276, 'description': 'Winst & risico'}, 'direct_general_costs': {'description': 'Totale directe algemene kosten', 'value_excl_BTW': 9611.548796929463, 'value_incl_BTW': 11629.974044284649}, 'indirect_general_costs': {'description': 'Totale indirecte algemene kosten', 'value_excl_BTW': 1217.5141092046488, 'value_incl_BTW': 1473.192072137625}, 'total_general_costs': {'description': 'Totale algemene kosten', 'value_excl_BTW': 10829.062906134111, 'value_incl_BTW': 13103.166116422273}}, 'Risicoreservering': {'code': '', 'surcharge_percentage': 10.0, 'base_cost': 151142.5525304272, 'value': 15114.25525304272, 'description': 'Objectoverstijgende risicoreservering (makkelijk)'}, 'Vastgoedkosten': {'direct_benoemd_real_estate_cost': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 700000.0, 'quantity': 0, 'unit': 'panden', 'description': '', 'dimension': None}, 'direct_niet_benoemd_real_estate_cost': {'code': 'Q-GVMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Nader te detailleren directe bouwkosten'}, 'indirect_real_estate_cost': {'code': 'Q-GVMAKIND', 'surcharge_percentage': 7.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Indirecte vastgoedkosten'}, 'real_estate_risk_cost': {'code': 'Q-GVMAKNBO', 'surcharge_percentage': 15.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Niet benoemd objectrisico vastgoed'}, 'total_real_estate_costs': {'description': 'Totale vastgoedkosten', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}}

    # --- Full recursive comparison ---
    for category, expected_items in EXPECTED_COST_DECOMPOSITION.items():
        assert category in costs, f"missing top-level category '{category}'"
        assert_cost_structure(
            costs[category],
            expected_items,
            path=category,
        )
