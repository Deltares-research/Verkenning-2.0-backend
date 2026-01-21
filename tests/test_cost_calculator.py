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
    costs = calculator_easy.calc_all_construction_costs(1000, 0)

    assert costs.indirect_costs > 0
    assert costs.total_costs > costs.direct_costs
    assert costs.total_costs == pytest.approx(1365.526464)

def test_construction_cost_medium(calculator_medium):
    costs = calculator_medium.calc_all_construction_costs(1000, 0)

    assert costs.indirect_costs > 0
    assert costs.total_costs > costs.direct_costs
    assert costs.total_costs == pytest.approx(1392.5665920000001)

def test_construction_cost_hard(calculator_hard):
    costs = calculator_hard.calc_all_construction_costs(1000, 0)

    assert costs.indirect_costs > 0
    assert costs.total_costs > costs.direct_costs
    assert costs.total_costs == pytest.approx(1419.60672)




