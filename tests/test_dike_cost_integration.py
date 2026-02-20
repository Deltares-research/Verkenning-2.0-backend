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

    EXPECTED_COST_DECOMPOSITION = {'Bouwkosten': {'Directe Bouwkosten': {'Directe kosten grondwerk': {'kosten_opruimen': {'value': 820.8411496489607, 'value_incl_BTW': 993.2177910752424, 'unit_cost': 0.28, 'quantity': 2931.5755344605736, 'unit': 'm2', 'description': 'Opruimen terrein en afvoeren naar stort', 'dimension': None}, 'kosten_maaien': {'value': 58.631510689211474, 'value_incl_BTW': 70.94412793394588, 'unit_cost': 0.02, 'quantity': 2931.5755344605736, 'unit': 'm2', 'description': 'Maaien terreinen en afvoeren maaisel naar stort', 'dimension': None}, 'afgraven_toplaag': {'value': 2175.2290465697456, 'value_incl_BTW': 2632.0271463493923, 'unit_cost': 3.71, 'quantity': 586.3151068921147, 'unit': 'm3', 'description': 'Ontgraven teelaarde en in tijdelijk depot zetten', 'dimension': None}, 'afgraven_oud_materiaal': {'value': 7199.9495126351685, 'value_incl_BTW': 8711.938910288554, 'unit_cost': 3.07, 'quantity': 2345.260427568459, 'unit': 'm3', 'description': 'Ontgraven zand en in tijdelijk depot zetten', 'dimension': None}, 'hergebruik_oud_materiaal': {'value': 6801.25523994853, 'value_incl_BTW': 8229.518840337721, 'unit_cost': 2.9, 'quantity': 2345.260427568459, 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen van zand (kern)', 'dimension': None}, 'aanvullen_kern': {'value': 20019.064945666425, 'value_incl_BTW': 24223.068584256373, 'unit_cost': 14.54, 'quantity': 1376.8270251489976, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) zand (kern)', 'dimension': None}, 'profileren_dijkkern': {'value': 2188.1033046948514, 'value_incl_BTW': 2647.60499868077, 'unit_cost': 0.73, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Profileren dijkprofiel kern', 'dimension': None}, 'aanbrengen_nieuwe_kleilaag': {'value': 40786.038036496066, 'value_incl_BTW': 49351.10602416024, 'unit_cost': 21.06, 'quantity': 1936.658976091931, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) klei', 'dimension': None}, 'profileren_nieuwe_kleilaag': {'value': 2397.921429802577, 'value_incl_BTW': 2901.4849300611186, 'unit_cost': 0.8, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Profileren dijkprofiel afdeklaag', 'dimension': None}, 'hergebruik_toplaag': {'value': 2304.218370086011, 'value_incl_BTW': 2788.104227804073, 'unit_cost': 3.93, 'quantity': 586.3151068921147, 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen toplaag (teelaarde)', 'dimension': None}, 'aanvullen_toplaag': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 16.9, 'quantity': 0, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) teelaarde', 'dimension': None}, 'profileren_nieuwe_toplaag': {'value': 2307.9993761849805, 'value_incl_BTW': 2792.6792451838264, 'unit_cost': 0.77, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Profileren dijkprofiel teelaarde (eindprofiel)', 'dimension': None}, 'inzaaien_nieuwe_toplaag': {'value': 1019.1166076660953, 'value_incl_BTW': 1233.1310952759752, 'unit_cost': 0.34, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Inzaaien dijkprofiel', 'dimension': None}, 'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk', 'value_excl_BTW': 88078.3685300886, 'value_incl_BTW': 106574.82592140725}}, 'Directe kosten constructies': {'directe_bouwkosten': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 0.0, 'quantity': 0.0, 'unit': '', 'description': '', 'dimension': None}, 'totale_BDBK_constructie': {'description': 'Benoemde directe bouwkosten constructie', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}, 'Directe kosten infrastructuur': {'verwijderen_weg': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 15.0, 'quantity': 0, 'unit': 'm2', 'description': 'Opbreken en afvoeren regionale weg (B=4-7m) (incl. stort-/recyclingskosten)', 'dimension': None}, 'aanleggen_weg': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 54.79, 'quantity': 0, 'unit': 'm2', 'description': 'Leveren en aanbrengen regionale weg (B=4-7m) exclusief bebording verlichting bermen en sloten', 'dimension': None}, 'verwijderen_fietspad': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 10.84, 'quantity': 0, 'unit': 'm2', 'description': 'Opbreken en afvoeren fietspad (B<2m) (incl. stort-/recyclingskosten)', 'dimension': None}, 'aanleggen_fietspad': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 38.61, 'quantity': 0, 'unit': 'm2', 'description': 'Leveren en aanbrengen fietspad (B<2m) exclusief bebording en verlichting', 'dimension': None}}}, 'Indirecte Bouwkosten': {'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk', 'value_excl_BTW': 88078.3685300886, 'value_incl_BTW': 106574.82592140725}, 'totale_BDBK_constructie': {'description': 'Benoemde directe bouwkosten constructie', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}, 'totale_BDBK_infrastructuur': {'description': 'Benoemde directe bouwkosten infrastructuur', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}, 'directe_niet_benoemde_bouwkosten_grondwerk': {'code': 'Q-GGMAKNTD', 'surcharge_percentage': 1.0, 'base_cost': 88078.3685300886, 'value': 880.783685300886, 'description': ''}, 'directe_niet_benoemde_bouwkosten_constructie': {'code': 'Q-GCMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': ''}, 'directe_niet_benoemde_bouwkosten_infrastructuur': {'code': 'Q-GCMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': ''}, 'pm_kosten': {'code': 'Q-EKABKUKMAN', 'surcharge_percentage': 20.0, 'base_cost': 88959.15221538949, 'value': 17791.8304430779, 'description': ''}, 'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 106750.98265846739, 'value': 7686.070751409652, 'description': ''}, 'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 114437.05340987704, 'value': 5836.289723903729, 'description': ''}, 'totale_directe_bouwkosten': {'description': 'Totale directe bouwkosten', 'value_excl_BTW': 88959.15221538949, 'value_incl_BTW': 107640.57418062132}, 'indirecte_bouwkosten': {'description': 'Indirecte bouwkosten', 'value_excl_BTW': 31314.190918391283, 'value_incl_BTW': 37890.17101125345}, 'totale_bouwkosten': {'description': 'Totale bouwkosten', 'value_excl_BTW': 120273.34313378077, 'value_incl_BTW': 145530.74519187477}}}, 'Engineeringkosten': {'engineering_opdrachtgever': {'code': 'Q-ENGOG1', 'surcharge_percentage': 8.0, 'base_cost': 120273.34313378077, 'value': 9621.867450702463, 'description': 'Engineeringskosten opdrachtgever (EPK) - makkelijk'}, 'engineering_opdrachtnemer': {'code': 'Q-ENGON1', 'surcharge_percentage': 5.9, 'base_cost': 120273.34313378077, 'value': 7096.127244893066, 'description': 'Engineeringskosten opdrachtnemer (schets-, voor-, definitief ontwerp, e.d.) - makkelijk'}, 'onderzoekskosten': {'code': 'Q-OND', 'surcharge_percentage': 1.0, 'base_cost': 120273.34313378077, 'value': 1202.7334313378078, 'description': 'Onderzoeken (archeologie, explosieven, LNC, e.d.))'}, 'algemene_kosten': {'code': '', 'surcharge_percentage': 7.199999999999999, 'base_cost': 17920.728126933336, 'value': 1290.2924251392, 'description': ''}, 'winst_en_risico': {'code': '', 'surcharge_percentage': 5.099999999999999, 'base_cost': 19211.020552072536, 'value': 979.7620481556992, 'description': ''}, 'direct_engineering_cost': {'description': 'Totale directe engineeringkosten', 'value_excl_BTW': 17920.728126933336, 'value_incl_BTW': 19663.48886894182}, 'indirect_engineering_cost': {'description': 'Totale indirecte engineeringkosten', 'value_excl_BTW': 2270.0544732948993, 'value_incl_BTW': 2490.8134620065975}, 'total_engineering_costs': {'description': 'Totale engineeringkosten', 'value_excl_BTW': 20190.782600228235, 'value_incl_BTW': 22154.302330948416}}, 'Overige bijkomende kosten': {'vergunningen_verzekeringen': {'code': 'Q-VERG', 'surcharge_percentage': 3.0, 'base_cost': 120273.34313378077, 'value': 3608.2002940134234, 'description': 'Vergunningen, heffingen en verzekeringen'}, 'kabels_leidingen': {'code': 'Q-KL', 'surcharge_percentage': 1.0, 'base_cost': 120273.34313378077, 'value': 1202.7334313378078, 'description': 'Kabels & leidingen'}, 'planschade_inpassingsmaatregelen': {'code': 'Q-PLAN', 'surcharge_percentage': 4.0, 'base_cost': 120273.34313378077, 'value': 4810.933725351231, 'description': 'Planschade & inpassingsmaatregelen'}, 'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 9621.867450702463, 'value': 692.7744564505773, 'description': 'Algemene kosten (AK)'}, 'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 10314.64190715304, 'value': 526.046737264805, 'description': 'Winst & risico'}, 'direct_general_costs': {'description': 'Totale directe algemene kosten', 'value_excl_BTW': 9621.867450702463, 'value_incl_BTW': 11642.459615349979}, 'indirect_general_costs': {'description': 'Totale indirecte algemene kosten', 'value_excl_BTW': 1218.8211937153824, 'value_incl_BTW': 1474.7736443956123}, 'total_general_costs': {'description': 'Totale algemene kosten', 'value_excl_BTW': 10840.688644417845, 'value_incl_BTW': 13117.233259745592}}, 'Risicoreservering': {'code': '', 'surcharge_percentage': 10.0, 'base_cost': 151304.81437842685, 'value': 15130.481437842685, 'description': 'Objectoverstijgende risicoreservering (makkelijk)'}, 'Vastgoedkosten': {'direct_benoemd_real_estate_cost': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 700000.0, 'quantity': 0, 'unit': 'panden', 'description': '', 'dimension': None}, 'direct_niet_benoemd_real_estate_cost': {'code': 'Q-GVMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Nader te detailleren directe bouwkosten'}, 'indirect_real_estate_cost': {'code': 'Q-GVMAKIND', 'surcharge_percentage': 7.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Indirecte vastgoedkosten'}, 'real_estate_risk_cost': {'code': 'Q-GVMAKNBO', 'surcharge_percentage': 15.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Niet benoemd objectrisico vastgoed'}, 'total_real_estate_costs': {'description': 'Totale vastgoedkosten', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}}


    # --- Full recursive comparison ---
    for category, expected_items in EXPECTED_COST_DECOMPOSITION.items():
        assert category in costs, f"missing top-level category '{category}'"
        assert_cost_structure(
            costs[category],
            expected_items,
            path=category,
        )
