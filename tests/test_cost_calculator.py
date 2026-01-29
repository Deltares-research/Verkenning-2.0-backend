from pathlib import Path

import pytest
import numpy as np
from app.cost_calculator import CostCalculator, EnumerationComplexity
from app.unit_costs_and_surcharges import load_kosten_catalogus



# --- Fixtures ----------------------------------------------------

@pytest.fixture(scope="session")
def catalogue():
    path_cost = Path(__file__).parent.joinpath("datasets/eenheidsprijzen.json")
    path_opslag_factor = Path(__file__).parent.joinpath("datasets/opslagfactoren.json")

    return load_kosten_catalogus(
        eenheidsprijzen=str(path_cost),
        opslagfactoren=str(path_opslag_factor),
    )

@pytest.fixture
def calculator_easy(catalogue):
    return CostCalculator(catalogue, "makkelijke maatregel")

@pytest.fixture
def calculator_medium(catalogue):
    return CostCalculator(catalogue, "gemiddelde maatregel")

@pytest.fixture
def calculator_hard(catalogue):
    return CostCalculator(catalogue, "moeilijke maatregel")

# --- Tests -------------------------------------------------------

def test_complexity_parsing():
    assert EnumerationComplexity.from_string("makkelijke maatregel") == EnumerationComplexity.EASY
    assert EnumerationComplexity.from_string("gemiddelde maatregel") == EnumerationComplexity.MEDIUM
    assert EnumerationComplexity.from_string("moeilijke maatregel") == EnumerationComplexity.HARD

    with pytest.raises(ValueError):
        EnumerationComplexity.from_string("unknown")


def test_construction_cost_ground_easy(calculator_easy):
    costs = calculator_easy.calc_construction_costs_groundwork(1000)

    assert costs.indirecte_bouwkosten > 0
    assert costs.totale_bouwkosten > costs.totale_BDBK_grondwerk
    assert costs.totale_bouwkosten == pytest.approx(1365.526464)

def test_construction_cost_medium(calculator_medium):
    costs = calculator_medium.calc_construction_costs_groundwork(1000)

    assert costs.indirecte_bouwkosten > 0
    assert costs.totale_bouwkosten > costs.totale_BDBK_grondwerk
    assert costs.totale_bouwkosten == pytest.approx(1392.5665920000001)

def test_construction_cost_ground_hard(calculator_hard):
    costs = calculator_hard.calc_construction_costs_groundwork(1000)

    assert costs.indirecte_bouwkosten > 0
    assert costs.totale_bouwkosten > costs.totale_BDBK_grondwerk
    assert costs.totale_bouwkosten == pytest.approx(1419.60672)

def test_construction_cost_structure_hard(calculator_hard):
    costs = calculator_hard.calc_construction_costs_structure(1000)

    assert costs.indirecte_bouwkosten > 0
    assert costs.totale_bouwkosten > costs.totale_BDBK_grondwerk
    assert costs.totale_bouwkosten == pytest.approx(1554.80736)

def test_engineering_cost_easy(calculator_easy):
    constr_cost_ground = calculator_easy.calc_construction_costs_groundwork(1000).totale_bouwkosten
    constr_cost_structure = calculator_easy.calc_construction_costs_structure(1000).totale_bouwkosten
    costs_ground = calculator_easy.calc_all_engineering_costs(constr_cost_ground)
    costs_structure = calculator_easy.calc_all_engineering_costs(constr_cost_structure)

    assert costs_ground.total_engineering_costs == pytest.approx(229.23656440492343)
    assert costs_structure.total_engineering_costs == pytest.approx(238.31524022294016)

def test_engineering_cost_medium(calculator_medium):
    constr_cost_ground = calculator_medium.calc_construction_costs_groundwork(1000).totale_bouwkosten
    constr_cost_structure = calculator_medium.calc_construction_costs_structure(1000).totale_bouwkosten
    costs_ground = calculator_medium.calc_all_engineering_costs(constr_cost_ground)
    costs_structure = calculator_medium.calc_all_engineering_costs(constr_cost_structure)

    assert costs_ground.total_engineering_costs == pytest.approx(312.224191681023)
    assert costs_structure.total_engineering_costs == pytest.approx(333.4433115040051)
    assert calculator_medium.calc_all_engineering_costs(constr_cost_ground.__add__(constr_cost_structure)).total_engineering_costs == pytest.approx(645.6675031850281)

def test_engineering_cost_hard(calculator_hard):
    constr_cost_ground = calculator_hard.calc_construction_costs_groundwork(1000).totale_bouwkosten
    constr_cost_structure = calculator_hard.calc_construction_costs_structure(1000).totale_bouwkosten
    costs_ground = calculator_hard.calc_all_engineering_costs(constr_cost_ground)
    costs_structure = calculator_hard.calc_all_engineering_costs(constr_cost_structure)

    assert costs_ground.total_engineering_costs == pytest.approx(398.25835446652417)
    assert costs_structure.total_engineering_costs == pytest.approx(436.187721558574)

def test_general_cost_easy(calculator_easy):
    constr_cost_ground = calculator_easy.calc_construction_costs_groundwork(1000).totale_bouwkosten
    constr_cost_structure = calculator_easy.calc_construction_costs_structure(1000).totale_bouwkosten
    costs_ground = calculator_easy.calc_general_costs(constr_cost_ground)
    costs_structure = calculator_easy.calc_general_costs(constr_cost_structure)

    assert costs_ground.total_general_costs == pytest.approx(123.08003457982463)
    assert costs_structure.total_general_costs == pytest.approx(127.95449139486722)
    np.testing.assert_almost_equal(costs_ground.total_general_costs/constr_cost_ground, costs_structure.total_general_costs/constr_cost_structure, decimal=3)

def test_general_cost_medium(calculator_medium):
    constr_cost_ground = calculator_medium.calc_construction_costs_groundwork(1000).totale_bouwkosten
    constr_cost_structure = calculator_medium.calc_construction_costs_structure(1000).totale_bouwkosten
    costs_ground = calculator_medium.calc_general_costs(constr_cost_ground)
    costs_structure = calculator_medium.calc_general_costs(constr_cost_structure)

    assert costs_ground.total_general_costs == pytest.approx(125.51726298734592)
    assert costs_structure.total_general_costs == pytest.approx(134.0475624136704)
    assert costs_ground.total_general_costs/constr_cost_ground == costs_structure.total_general_costs/constr_cost_structure

def test_general_cost_hard(calculator_hard):
    constr_cost_ground = calculator_hard.calc_construction_costs_groundwork(1000).totale_bouwkosten
    constr_cost_structure = calculator_hard.calc_construction_costs_structure(1000).totale_bouwkosten

    costs_ground = calculator_hard.calc_general_costs(constr_cost_ground)
    costs_structure = calculator_hard.calc_general_costs(constr_cost_structure)

    assert costs_ground.total_general_costs == pytest.approx(127.9544913948672)
    assert costs_structure.total_general_costs == pytest.approx(140.1406334324736)
    assert costs_ground.total_general_costs/constr_cost_ground == costs_structure.total_general_costs/constr_cost_structure

def test_total_cost_ground_easy(calculator_easy):
    constr_cost_ground = calculator_easy.calc_construction_costs_groundwork(1000).totale_bouwkosten
    eng_cost = calculator_easy.calc_all_engineering_costs(constr_cost_ground).total_engineering_costs
    gen_cost = calculator_easy.calc_general_costs(constr_cost_ground).total_general_costs
    total_investment = constr_cost_ground + eng_cost + gen_cost
    risk_cost = calculator_easy.calc_risk_cost(total_investment, grond_percentage=1.0)

    total_cost_excl_BTW = total_investment + risk_cost

    assert total_investment == pytest.approx(1717.84306)
    assert risk_cost == pytest.approx(171.7843062)
    assert total_cost_excl_BTW == pytest.approx(1889.6273692832228)

def test_total_cost_ground_medium(calculator_medium):
    constr_cost_ground = calculator_medium.calc_construction_costs_groundwork(1000).totale_bouwkosten
    eng_cost = calculator_medium.calc_all_engineering_costs(constr_cost_ground).total_engineering_costs
    gen_cost = calculator_medium.calc_general_costs(constr_cost_ground).total_general_costs
    total_investment = constr_cost_ground + eng_cost + gen_cost
    risk_cost = calculator_medium.calc_risk_cost(total_investment, grond_percentage=1.0)

    total_cost_excl_BTW = total_investment + risk_cost

    assert total_investment == pytest.approx(1830.3080466683691)
    assert risk_cost == pytest.approx(274.5462070002554)
    assert total_cost_excl_BTW == pytest.approx(2104.8542536686246)


def test_total_cost_ground_hard(calculator_hard):
    constr_cost_ground = calculator_hard.calc_construction_costs_groundwork(1000).totale_bouwkosten
    eng_cost = calculator_hard.calc_all_engineering_costs(constr_cost_ground).total_engineering_costs
    gen_cost = calculator_hard.calc_general_costs(constr_cost_ground).total_general_costs
    total_investment = constr_cost_ground + eng_cost + gen_cost
    risk_cost = calculator_hard.calc_risk_cost(total_investment, grond_percentage=1.0)
    total_cost_excl_BTW = total_investment + risk_cost

    assert total_investment == pytest.approx(1945.8195658613915)
    assert risk_cost == pytest.approx(389.16391317227834)
    assert total_cost_excl_BTW == pytest.approx(2334.98347903367)

def test_total_cost_medium_both(calculator_medium):
    constr_cost_ground = calculator_medium.calc_construction_costs_groundwork(500).totale_bouwkosten
    constr_cost_structure = calculator_medium.calc_construction_costs_structure(500).totale_bouwkosten
    eng_cost = calculator_medium.calc_all_engineering_costs(constr_cost_ground+constr_cost_structure).total_engineering_costs
    gen_cost = calculator_medium.calc_general_costs(constr_cost_ground+constr_cost_structure).total_general_costs
    total_investment = constr_cost_ground + constr_cost_structure + eng_cost + gen_cost
    grond_percentage = constr_cost_ground / (constr_cost_ground + constr_cost_structure)
    assert grond_percentage == pytest.approx(0.4835680751173709)
    risk_cost = calculator_medium.calc_risk_cost(total_investment, grond_percentage=grond_percentage)

    total_cost_excl_BTW = total_investment + risk_cost

    assert total_investment == pytest.approx(1892.5029802930221)
    assert risk_cost == pytest.approx(381.61034273983717)
    assert total_cost_excl_BTW == pytest.approx(2274.1133230328596)
