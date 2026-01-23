from pathlib import Path

import pytest
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


def test_construction_cost_easy(calculator_easy):
    costs = calculator_easy.calc_all_construction_costs(1000)

    assert costs.indirect_costs > 0
    assert costs.total_costs > costs.direct_costs
    assert costs.total_costs == pytest.approx(1365.526464)

def test_construction_cost_medium(calculator_medium):
    costs = calculator_medium.calc_all_construction_costs(1000)

    assert costs.indirect_costs > 0
    assert costs.total_costs > costs.direct_costs
    assert costs.total_costs == pytest.approx(1392.5665920000001)

def test_construction_cost_hard(calculator_hard):
    costs = calculator_hard.calc_all_construction_costs(1000)

    assert costs.indirect_costs > 0
    assert costs.total_costs > costs.direct_costs
    assert costs.total_costs == pytest.approx(1419.60672)

def test_engineering_cost_easy(calculator_easy):
    constr_cost = calculator_easy.calc_all_construction_costs(1000).total_costs
    costs = calculator_easy.calc_all_engineering_costs(constr_cost)

    assert costs.total_engineering_costs == pytest.approx(229.23656440492343)

def test_engineering_cost_medium(calculator_medium):
    constr_cost = calculator_medium.calc_all_construction_costs(1000).total_costs
    costs = calculator_medium.calc_all_engineering_costs(constr_cost)

    assert costs.total_engineering_costs == pytest.approx(312.224191681023)

def test_engineering_cost_hard(calculator_hard):
    constr_cost = calculator_hard.calc_all_construction_costs(1000).total_costs
    costs = calculator_hard.calc_all_engineering_costs(constr_cost)

    assert costs.total_engineering_costs == pytest.approx(398.25835446652417)

def test_general_cost_easy(calculator_easy):
    constr_cost = calculator_easy.calc_all_construction_costs(1000).total_costs
    costs = calculator_easy.calc_general_costs(constr_cost)

    assert costs.total_general_costs == pytest.approx(123.08003457982463)

def test_general_cost_medium(calculator_medium):
    constr_cost = calculator_medium.calc_all_construction_costs(1000).total_costs
    costs = calculator_medium.calc_general_costs(constr_cost)

    assert costs.total_general_costs == pytest.approx(125.51726298734592)

def test_general_cost_hard(calculator_hard):
    constr_cost = calculator_hard.calc_all_construction_costs(1000).total_costs
    costs = calculator_hard.calc_general_costs(constr_cost)

    assert costs.total_general_costs == pytest.approx(127.9544913948672)

def test_total_cost_easy(calculator_easy):
    constr_cost = calculator_easy.calc_all_construction_costs(1000).total_costs
    eng_cost = calculator_easy.calc_all_engineering_costs(constr_cost).total_engineering_costs
    gen_cost = calculator_easy.calc_general_costs(constr_cost).total_general_costs
    total_investment = constr_cost + eng_cost + gen_cost
    risk_cost = calculator_easy.calc_risk_cost(total_investment)

    total_cost_excl_BTW = total_investment + risk_cost

    assert total_investment == pytest.approx(1717.84306)
    assert risk_cost == pytest.approx(171.7843062)
    assert total_cost_excl_BTW == pytest.approx(1889.6273692832228)

def test_total_cost_medium(calculator_medium):
    constr_cost = calculator_medium.calc_all_construction_costs(1000).total_costs
    eng_cost = calculator_medium.calc_all_engineering_costs(constr_cost).total_engineering_costs
    gen_cost = calculator_medium.calc_general_costs(constr_cost).total_general_costs
    total_investment = constr_cost + eng_cost + gen_cost
    risk_cost = calculator_medium.calc_risk_cost(total_investment)

    total_cost_excl_BTW = total_investment + risk_cost

    assert total_investment == pytest.approx(1830.3080466683691)
    assert risk_cost == pytest.approx(274.5462070002554)
    assert total_cost_excl_BTW == pytest.approx(2104.8542536686246)

def test_total_cost_hard(calculator_hard):
    constr_cost = calculator_hard.calc_all_construction_costs(1000).total_costs
    eng_cost = calculator_hard.calc_all_engineering_costs(constr_cost).total_engineering_costs
    gen_cost = calculator_hard.calc_general_costs(constr_cost).total_general_costs
    total_investment = constr_cost + eng_cost + gen_cost
    risk_cost = calculator_hard.calc_risk_cost(total_investment)

    total_cost_excl_BTW = total_investment + risk_cost

    assert total_investment == pytest.approx(1945.8195658613915)
    assert risk_cost == pytest.approx(389.16391317227834)
    assert total_cost_excl_BTW == pytest.approx(2334.98347903367)