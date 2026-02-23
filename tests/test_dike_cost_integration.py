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

    EXPECTED_COST_DECOMPOSITION = {'Bouwkosten': {'Directe Bouwkosten': {'Directe kosten grondwerk': {'kosten_opruimen': {'value': 820.8537722873807, 'value_incl_BTW': 993.2330644677306, 'unit_cost': 0.28, 'quantity': 2931.620615312074, 'unit': 'm2', 'description': 'Opruimen terrein en afvoeren naar stort', 'dimension': None}, 'kosten_maaien': {'value': 58.63241230624148, 'value_incl_BTW': 70.94521889055218, 'unit_cost': 0.02, 'quantity': 2931.620615312074, 'unit': 'm2', 'description': 'Maaien terreinen en afvoeren maaisel naar stort', 'dimension': None}, 'afgraven_toplaag': {'value': 2172.935046083261, 'value_incl_BTW': 2629.251405760746, 'unit_cost': 3.71, 'quantity': 585.6967779200165, 'unit': 'm3', 'description': 'Ontgraven teelaarde en in tijdelijk depot zetten', 'dimension': None}, 'afgraven_oud_materiaal': {'value': 7192.356432857802, 'value_incl_BTW': 8702.75128375794, 'unit_cost': 3.07, 'quantity': 2342.787111680066, 'unit': 'm3', 'description': 'Ontgraven zand en in tijdelijk depot zetten', 'dimension': None}, 'hergebruik_oud_materiaal': {'value': 2292.233341076139, 'value_incl_BTW': 2773.602342702128, 'unit_cost': 2.9, 'quantity': 790.4252900262549, 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen van zand (kern)', 'dimension': None}, 'aanvullen_kern': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 14.54, 'quantity': 0, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) zand (kern)', 'dimension': None}, 'profileren_dijkkern': {'value': 2153.506984185311, 'value_incl_BTW': 2605.7434508642264, 'unit_cost': 0.73, 'quantity': 2950.0095673771384, 'unit': 'm2', 'description': 'Profileren dijkprofiel kern', 'dimension': None}, 'aanbrengen_nieuwe_kleilaag': {'value': 40793.81767998092, 'value_incl_BTW': 49360.519392776914, 'unit_cost': 21.06, 'quantity': 1937.0283798661408, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) klei', 'dimension': None}, 'profileren_nieuwe_kleilaag': {'value': 2360.007653901711, 'value_incl_BTW': 2855.60926122107, 'unit_cost': 0.8, 'quantity': 2950.0095673771384, 'unit': 'm2', 'description': 'Profileren dijkprofiel afdeklaag', 'dimension': None}, 'hergebruik_toplaag': {'value': 2301.788337225665, 'value_incl_BTW': 2785.1638880430546, 'unit_cost': 3.93, 'quantity': 585.6967779200165, 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen toplaag (teelaarde)', 'dimension': None}, 'aanvullen_toplaag': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 16.9, 'quantity': 0, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) teelaarde', 'dimension': None}, 'profileren_nieuwe_toplaag': {'value': 2307.9993761849805, 'value_incl_BTW': 2792.6792451838264, 'unit_cost': 0.77, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Profileren dijkprofiel teelaarde (eindprofiel)', 'dimension': None}, 'inzaaien_nieuwe_toplaag': {'value': 1019.1166076660953, 'value_incl_BTW': 1233.1310952759752, 'unit_cost': 0.34, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Inzaaien dijkprofiel', 'dimension': None}, 'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk', 'value_excl_BTW': 63473.24764375551, 'value_incl_BTW': 76802.62964894416}}, 'Directe kosten constructies': {'directe_bouwkosten': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 0.0, 'quantity': 0.0, 'unit': '', 'description': '', 'dimension': None}, 'totale_BDBK_constructie': {'description': 'Benoemde directe bouwkosten constructie', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}, 'Directe kosten infrastructuur': {'verwijderen_weg': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 15.0, 'quantity': 0, 'unit': 'm2', 'description': 'Opbreken en afvoeren regionale weg (B=4-7m) (incl. stort-/recyclingskosten)', 'dimension': None}, 'aanleggen_weg': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 54.79, 'quantity': 0, 'unit': 'm2', 'description': 'Leveren en aanbrengen regionale weg (B=4-7m) exclusief bebording verlichting bermen en sloten', 'dimension': None}, 'verwijderen_fietspad': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 10.84, 'quantity': 0, 'unit': 'm2', 'description': 'Opbreken en afvoeren fietspad (B<2m) (incl. stort-/recyclingskosten)', 'dimension': None}, 'aanleggen_fietspad': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 38.61, 'quantity': 0, 'unit': 'm2', 'description': 'Leveren en aanbrengen fietspad (B<2m) exclusief bebording en verlichting', 'dimension': None}}}, 'Indirecte Bouwkosten': {'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk', 'value_excl_BTW': 63473.24764375551, 'value_incl_BTW': 76802.62964894416}, 'totale_BDBK_constructie': {'description': 'Benoemde directe bouwkosten constructie', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}, 'totale_BDBK_infrastructuur': {'description': 'Benoemde directe bouwkosten infrastructuur', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}, 'directe_niet_benoemde_bouwkosten_grondwerk': {'code': 'Q-GGMAKNTD', 'surcharge_percentage': 1.0, 'base_cost': 63473.24764375551, 'value': 634.7324764375551, 'description': ''}, 'directe_niet_benoemde_bouwkosten_constructie': {'code': 'Q-GCMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': ''}, 'directe_niet_benoemde_bouwkosten_infrastructuur': {'code': 'Q-GCMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': ''}, 'pm_kosten': {'code': 'Q-EKABKUKMAN', 'surcharge_percentage': 20.0, 'base_cost': 64107.98012019306, 'value': 12821.596024038612, 'description': ''}, 'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 76929.57614423167, 'value': 5538.92948238468, 'description': ''}, 'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 82468.50562661636, 'value': 4205.893786957434, 'description': ''}, 'totale_directe_bouwkosten': {'description': 'Totale directe bouwkosten', 'value_excl_BTW': 64107.98012019306, 'value_incl_BTW': 77570.6559454336}, 'indirecte_bouwkosten': {'description': 'Indirecte bouwkosten', 'value_excl_BTW': 22566.419293380724, 'value_incl_BTW': 27305.36734499068}, 'totale_bouwkosten': {'description': 'Totale bouwkosten', 'value_excl_BTW': 86674.39941357379, 'value_incl_BTW': 104876.02329042427}}}, 'Engineeringkosten': {'engineering_opdrachtgever': {'code': 'Q-ENGOG1', 'surcharge_percentage': 8.0, 'base_cost': 86674.39941357379, 'value': 6933.951953085903, 'description': 'Engineeringskosten opdrachtgever (EPK) - makkelijk'}, 'engineering_opdrachtnemer': {'code': 'Q-ENGON1', 'surcharge_percentage': 5.9, 'base_cost': 86674.39941357379, 'value': 5113.7895654008535, 'description': 'Engineeringskosten opdrachtnemer (schets-, voor-, definitief ontwerp, e.d.) - makkelijk'}, 'onderzoekskosten': {'code': 'Q-OND', 'surcharge_percentage': 1.0, 'base_cost': 86674.39941357379, 'value': 866.7439941357379, 'description': 'Onderzoeken (archeologie, explosieven, LNC, e.d.))'}, 'algemene_kosten': {'code': '', 'surcharge_percentage': 7.200000000000001, 'base_cost': 12914.485512622494, 'value': 929.8429569088198, 'description': ''}, 'winst_en_risico': {'code': '', 'surcharge_percentage': 5.1000000000000005, 'base_cost': 13844.328469531314, 'value': 706.0607519460972, 'description': ''}, 'direct_engineering_cost': {'description': 'Totale directe engineeringkosten', 'value_excl_BTW': 12914.485512622494, 'value_incl_BTW': 14170.397560125177}, 'indirect_engineering_cost': {'description': 'Totale indirecte engineeringkosten', 'value_excl_BTW': 1635.903708854917, 'value_incl_BTW': 1794.992599736177}, 'total_engineering_costs': {'description': 'Totale engineeringkosten', 'value_excl_BTW': 14550.389221477411, 'value_incl_BTW': 15965.390159861354}}, 'Overige bijkomende kosten': {'vergunningen_verzekeringen': {'code': 'Q-VERG', 'surcharge_percentage': 3.0, 'base_cost': 86674.39941357379, 'value': 2600.2319824072138, 'description': 'Vergunningen, heffingen en verzekeringen'}, 'kabels_leidingen': {'code': 'Q-KL', 'surcharge_percentage': 1.0, 'base_cost': 86674.39941357379, 'value': 866.7439941357379, 'description': 'Kabels & leidingen'}, 'planschade_inpassingsmaatregelen': {'code': 'Q-PLAN', 'surcharge_percentage': 4.0, 'base_cost': 86674.39941357379, 'value': 3466.9759765429517, 'description': 'Planschade & inpassingsmaatregelen'}, 'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 6933.951953085903, 'value': 499.24454062218507, 'description': 'Algemene kosten (AK)'}, 'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 7433.196493708088, 'value': 379.0930211791125, 'description': 'Winst & risico'}, 'direct_general_costs': {'description': 'Totale directe algemene kosten', 'value_excl_BTW': 6933.951953085903, 'value_incl_BTW': 8390.081863233943}, 'indirect_general_costs': {'description': 'Totale indirecte algemene kosten', 'value_excl_BTW': 878.3375618012976, 'value_incl_BTW': 1062.78844977957}, 'total_general_costs': {'description': 'Totale algemene kosten', 'value_excl_BTW': 7812.289514887201, 'value_incl_BTW': 9452.870313013513}}, 'Risicoreservering': {'code': '', 'surcharge_percentage': 10.0, 'base_cost': 109037.0781499384, 'value': 10903.707814993839, 'description': 'Objectoverstijgende risicoreservering (makkelijk)'}, 'Vastgoedkosten': {'direct_benoemd_real_estate_cost': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 700000.0, 'quantity': 0, 'unit': 'panden', 'description': '', 'dimension': None}, 'direct_niet_benoemd_real_estate_cost': {'code': 'Q-GVMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Nader te detailleren directe bouwkosten'}, 'indirect_real_estate_cost': {'code': 'Q-GVMAKIND', 'surcharge_percentage': 7.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Indirecte vastgoedkosten'}, 'real_estate_risk_cost': {'code': 'Q-GVMAKNBO', 'surcharge_percentage': 15.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Niet benoemd objectrisico vastgoed'}, 'total_real_estate_costs': {'description': 'Totale vastgoedkosten', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}}

    # --- Full recursive comparison ---
    for category, expected_items in EXPECTED_COST_DECOMPOSITION.items():
        assert category in costs, f"missing top-level category '{category}'"
        assert_cost_structure(
            costs[category],
            expected_items,
            path=category,
        )
