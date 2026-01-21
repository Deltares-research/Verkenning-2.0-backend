from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict



@dataclass
class SurchargeItem:
    code: str
    price_percent: float  # renamed from prijs to make it clear it's a percentage


@dataclass
class ConstructionCosts:
    preparation: float  # Voorbereiding
    groundwork: float  # Benoemde Directe BouwKosten (BDBK)
    direct_costs: float # Directe bouwkosten (DBK)
    pm_cost: float
    general_cost: float
    risk_profit: float
    indirect_costs: float # Indirecte bouwkosten (IBK)
    total_costs: float

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        return asdict(self)

@dataclass
class EngineeringCosts:
    epk_cost: float  # Engineeringskosten opdrachtgever (EPK)
    design_cost: float  # Engineeringskosten opdrachtnemer (schets-, voor-, definitief ontwerp, e.d.)
    research_cost: float  # Onderzoeken (archeologie, explosieven, LNC, e.d.))
    direct_engineering_cost: float
    general_cost: float
    risk_profit: float
    indirect_engineering_costs: float
    total_engineering_costs: float

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        return asdict(self)

@dataclass
class GeneralCosts:
    insurances: float # Vergunningen, heffingen en verzekeringen
    cables_pipes: float # Kabels & leidingen
    damages: float # Planschade & inpassingsmaatregelen
    direct_general_costs: float
    general_cost: float
    risk_profit: float
    indirect_general_costs: float
    total_general_costs: float

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        return asdict(self)

class EnumerationComplexity(Enum):
    EASY = 'makkelijke maatregel'
    MEDIUM = 'gemiddelde maatregel'
    HARD = 'moeilijke maatregel'


    @classmethod
    def from_string(cls, value: str) -> "EnumerationComplexity":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown complexity: {value}")

class CostCalculator:
    def __init__(self, catalogue, complexity: str):
        """
        Expects catalogue.categorieen['Percentages ter bepaling Opslagfactor investeringskosten / benoemde directe bouwkosten algemeen']
        to be a list of items with .code and .prijs attributes
        """
        self.complexity = EnumerationComplexity.from_string(complexity)

        categories = [
            'Percentages ter bepaling Opslagfactor investeringskosten / benoemde directe bouwkosten algemeen',
            'Percentages ter bepaling Opslagfactor investeringskosten / benoemde directe bouwkosten Grondversterkingen',
            'Percentages ter bepaling Opslagfactor investeringskosten / benoemde directe bouwkosten Constructieve versterkingen',
        ]

        all_items = []
        for cat in categories:
            all_items.extend(catalogue.categorieen[cat])

        self.surcharge_dict: Dict[str, SurchargeItem] = {
            item.code: SurchargeItem(item.code, item.prijs)
            for item in all_items
        }

    def calc_all_construction_costs(self, groundwork_cost: float, preparation_cost: float) -> ConstructionCosts:

        if self.complexity == EnumerationComplexity.EASY:
            directe_bouwkosten = groundwork_cost * self.surcharge_dict['Q-GGMAKNTD'].price_percent / 100
        elif self.complexity == EnumerationComplexity.MEDIUM:
            directe_bouwkosten = groundwork_cost * self.surcharge_dict['Q-GGGEMNTD'].price_percent / 100
        elif self.complexity == EnumerationComplexity.HARD:
            directe_bouwkosten = groundwork_cost * self.surcharge_dict['Q-GGMOENTD'].price_percent / 100
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")

        # Helper to calculate surcharge costs
        def surcharge(code: str) -> float:
            return directe_bouwkosten * self.surcharge_dict[code].price_percent / 100.0

        pm_cost = surcharge("Q-EKABKUKMAN")  # Project management etc.
        general_cost = surcharge("Q-AK")     # Algemene kosten
        risk_profit = surcharge("Q-WR")      # Winst & risico

        indirecte_bouwkosten = pm_cost + general_cost + risk_profit
        total_costs = directe_bouwkosten + indirecte_bouwkosten

        return ConstructionCosts(
            preparation=preparation_cost,
            groundwork=groundwork_cost,
            direct_costs=directe_bouwkosten,
            pm_cost=pm_cost,
            general_cost=general_cost,
            risk_profit=risk_profit,
            indirect_costs=indirecte_bouwkosten,
            total_costs=total_costs,
        )


    def calc_all_engineering_costs(self, construction_cost: float) -> EngineeringCosts:
        """

        :param construction_cost: Total construction cost from calc_all_construction_costs
        """

        if self.complexity == EnumerationComplexity.EASY:
            epk_cost = construction_cost * self.surcharge_dict["Q-ENGOG1"].price_percent / 100.0
            design_cost = construction_cost * self.surcharge_dict["Q-ENGON1"].price_percent / 100.0
        elif self.complexity == EnumerationComplexity.MEDIUM:
            epk_cost = construction_cost * self.surcharge_dict["Q-ENGOG2"].price_percent / 100.0
            design_cost = construction_cost * self.surcharge_dict["Q-ENGON2"].price_percent / 100.0
        elif self.complexity == EnumerationComplexity.HARD:
            epk_cost = construction_cost * self.surcharge_dict["Q-ENGOG3"].price_percent / 100.0
            design_cost = construction_cost * self.surcharge_dict["Q-ENGON3"].price_percent / 100.0
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")
        research_cost = construction_cost * self.surcharge_dict["Q-OND"].price_percent / 100.0
        direct_engineering_cost = epk_cost + design_cost + research_cost


        # Helper to calculate surcharge costs
        def surcharge(code: str) -> float:
            return direct_engineering_cost * self.surcharge_dict[code].price_percent / 100.0

        general_cost = surcharge("Q-AK")     # Algemene kosten
        risk_profit = surcharge("Q-WR")      # Winst & risico

        indirect_engineering_cost = general_cost + risk_profit
        total_costs = direct_engineering_cost + indirect_engineering_cost

        return EngineeringCosts(
            epk_cost=epk_cost,
            design_cost=design_cost,
            research_cost=research_cost,
            direct_engineering_cost=direct_engineering_cost,
            general_cost=general_cost,
            risk_profit=risk_profit,
            indirect_engineering_costs=indirect_engineering_cost,
            total_engineering_costs=total_costs,
        )

    def calc_general_costs(self, construction_cost: float) -> GeneralCosts:

        insurances = construction_cost * self.surcharge_dict['Q-VERG'].price_percent / 100.0
        cables_pipes = construction_cost * self.surcharge_dict['Q-KL'].price_percent / 100.0
        damages = construction_cost * self.surcharge_dict['Q-PLAN'].price_percent / 100.0
        direct_general_costs = insurances + cables_pipes + damages

        genral_cost = direct_general_costs * self.surcharge_dict["Q-AK"].price_percent / 100.0
        risk_profit = direct_general_costs * self.surcharge_dict["Q-WR"].price_percent / 100.0

        indirect_general_costs = genral_cost + risk_profit
        total_general_costs = direct_general_costs + indirect_general_costs

        return GeneralCosts(
            insurances=insurances,
            cables_pipes=cables_pipes,
            damages=damages,
            direct_general_costs=direct_general_costs,
            general_cost=genral_cost,
            risk_profit=risk_profit,
            indirect_general_costs=indirect_general_costs,
            total_general_costs=total_general_costs,
        )

    def calc_risk_cost(self, investering_cost: float) -> float:
        if self.complexity == EnumerationComplexity.EASY:
            return investering_cost * self.surcharge_dict['Q-GGMAKONV'].price_percent / 100.0
        elif self.complexity == EnumerationComplexity.MEDIUM:
            return investering_cost * self.surcharge_dict['Q-GGGEMONV'].price_percent / 100.0
        elif self.complexity == EnumerationComplexity.HARD:
            return investering_cost * self.surcharge_dict['Q-GGMOEONV'].price_percent / 100.0
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")
