from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict


@dataclass
class SurchargeItem:
    code: str
    price_percent: float  # renamed from prijs to make it clear it's a percentage

@dataclass
class UnitPriceItem:
    code: str
    price: float  # renamed from prijs to make it clear it's a percentage


@dataclass
class DirectCostGroundWork:
    preparation_cost: float
    afgraven_grasbekleding_cost: float
    afgraven_kleilaag_cost: float
    herkeuren_kleilaag_cost: float
    aanvullen_kern_cost: float
    profieleren_dijkkern_cost: float
    aanbregen_nieuwe_kleilaag_cost: float
    profieleren_vannieuwe_kleilaag_cost: float
    hergebruik_teelaarde_cost: float
    aanvullen_teelaarde_cost: float
    profieleren_nieuwe_graslaag_cost: float

    @property
    def totale_BDBK_grondwerk(self) -> float:
        """Benoemde Directe BouwKosten (BDBK)"""
        return (
            self.preparation_cost +
            self.afgraven_grasbekleding_cost +
            self.afgraven_kleilaag_cost +
            self.herkeuren_kleilaag_cost +
            self.aanvullen_kern_cost +
            self.profieleren_dijkkern_cost +
            self.aanbregen_nieuwe_kleilaag_cost +
            self.profieleren_vannieuwe_kleilaag_cost +
            self.hergebruik_teelaarde_cost +
            self.aanvullen_teelaarde_cost +
            self.profieleren_nieuwe_graslaag_cost
        )

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        data = asdict(self)
        data['totale_BDBK_grondwerk'] = self.totale_BDBK_grondwerk
        return data


    @classmethod
    def zero(cls) -> "DirectCostGroundWork":
        return cls(**{field: 0.0 for field in cls.__dataclass_fields__})


@dataclass
class ConstructionCosts:
    totale_BDBK_grondwerk: float  # Benoemde Directe BouwKosten (BDBK) (deel grond)
    totale_BDBK_constructie: float # Benoemde Directe BouwKosten (BDBK) (deel constructies)
    totale_benoemde_directe_bouwkosten: float # Directe bouwkosten (DBK)
    pm_kosten: float
    algemene_kosten: float
    risico_en_winst: float
    indirecte_bouwkosten: float # Indirecte bouwkosten (IBK)
    totale_bouwkosten: float #Indirecte en directe bouwkosten (Totaal)

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
    
@dataclass
class RealEstateCosts:
    road_cost: float
    house_cost: float
    total_real_estate_costs: float

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        return asdict(self)

@dataclass
class StructureCosts:
    totale_BDBK_constructie: float

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        return asdict(self)
    
    @classmethod
    def zero(cls) -> "DirectCostGroundWork":
        return cls(**{field: 0.0 for field in cls.__dataclass_fields__})

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

        categories_surcharges = [
            'Percentages ter bepaling Opslagfactor investeringskosten / benoemde directe bouwkosten algemeen',
            'Percentages ter bepaling Opslagfactor investeringskosten / benoemde directe bouwkosten Grondversterkingen',
            'Percentages ter bepaling Opslagfactor investeringskosten / benoemde directe bouwkosten Constructieve versterkingen',
        ]

        categories_unit_prices = [
            'Grondverzet',
            'Profielafwerking',
            'Algemene werkzaamheden',
            'Wegen fietspaden en op-/afritten'
        ]

        all_items_surcharges = []
        for cat in categories_surcharges:
            all_items_surcharges.extend(catalogue.categorieen[cat])

        all_items_unit_prices = []
        for cat in categories_unit_prices:
            all_items_unit_prices.extend(catalogue.categorieen[cat])

        self.surcharge_dict: Dict[str, SurchargeItem] = {
            item.code: SurchargeItem(item.code, item.prijs)
            for item in all_items_surcharges
        }

        self.unit_price_dict: Dict[str, UnitPriceItem] = {
            item.code: UnitPriceItem(item.code, item.prijs)
            for item in all_items_unit_prices
        }

    def calc_real_estate_costs(self, nb_houses: float, road_area: float) -> RealEstateCosts:
        """
        Calculate real estate costs based on base cost and surcharge percentage.
        """
        ROAD_UNIT_COST = self.unit_price_dict['O-413'].price + self.unit_price_dict['O-513'].price
        COST_HOUSES = 700000
        return RealEstateCosts(
            road_cost=road_area * ROAD_UNIT_COST,
            house_cost=nb_houses * COST_HOUSES,
            total_real_estate_costs=road_area * ROAD_UNIT_COST + nb_houses * COST_HOUSES
        )

    def calc_direct_cost_ground_work(self, volumes: dict) -> DirectCostGroundWork:
        """
        Calculate the benoemde directe bouwkosten for ground work based on volumes and unit prices.
        """
        Q_GV010 = self.unit_price_dict['Q-GV010'].price
        Q_GV030 = self.unit_price_dict['Q-GV030'].price
        Q_GV050 = self.unit_price_dict['Q-GV050'].price
        Q_GV060 = self.unit_price_dict['Q-GV060'].price
        Q_GV070 = self.unit_price_dict['Q-GV070'].price
        Q_GV080 = self.unit_price_dict['Q-GV080'].price
        Q_GV090 = self.unit_price_dict['Q-GV090'].price
        Q_GV100 = self.unit_price_dict['Q-GV100'].price
        Q_GV110 = self.unit_price_dict['Q-GV110'].price
        Q_GV120 = self.unit_price_dict['Q-GV120'].price
        Q_AW010 = self.unit_price_dict['Q-AW010'].price
        Q_AW020 = self.unit_price_dict['Q-AW020'].price
        Q_AW030 = self.unit_price_dict['Q-AW030'].price


        V1b = volumes['V1b']  # Volume grasbekleding van het huidig profiel (verwijderd en hergebruikt)
        V2b = volumes['V2b']  # Volume kleilaag van het huidig profiel (verwijderd en hergebruikt als kernmateriaal)
        V3 = volumes['V3']  # volume grasbekleding van de nieuwe dijk
        V4 = volumes['V4']  # volume kleilaag van de nieuwe dijk
        V5 = volumes['V5']  # volume kernmateriaal van de nieuwe dijk
        S0 = volumes['S0']  # surface area beyond the toe of the old dike
        S5 = volumes['S5']  # surface area beyond the toe of the old dike

        ### Combine to get costs
        preparation_cost = S0 * (Q_AW010 + Q_AW020)  # Voorbereiden terrein
        afgraven_grasbekleding_cost = V1b * Q_GV010  # afgraven oude grasbekleding naar depot
        afgraven_kleilaag_cost = V2b * Q_GV030  # afgraven oude kleilaag naar depot
        herkeuren_kleilaag_cost = V2b * Q_GV050  # hergebruiken oude kleilaag in nieuwe kern
        aanvullen_kern_cost = (V5 + V1b) * Q_GV090  # aanvullen nieuwe kern met nieuw materiaal
        profieleren_dijkkern_cost = S5 * Q_GV100  # profieleren van dijkkern
        aanbregen_nieuwe_kleilaag_cost = V4 * Q_GV080  # aanbregen nieuwe kleilaag
        profieleren_vannieuwe_kleilaag_cost = S5 * Q_GV110  # profileren nieuwe kleilaar
        hergebruik_teelaarde_cost = V1b * Q_GV060  # hergebruiken teelaarde in nieuwe bekleding
        aanvullen_teelaarde_cost = (V3 - V1b) * Q_GV070  # aanvullen teelaarde in nieuwe bekleding
        profieleren_nieuwe_graslaag_cost = S5 * (Q_GV120 - Q_AW030)  # profileren nieuwe graslaag en inzaaien

        return DirectCostGroundWork(
            preparation_cost=preparation_cost,
            afgraven_grasbekleding_cost=afgraven_grasbekleding_cost,
            afgraven_kleilaag_cost=afgraven_kleilaag_cost,
            herkeuren_kleilaag_cost=herkeuren_kleilaag_cost,
            aanvullen_kern_cost=aanvullen_kern_cost,
            profieleren_dijkkern_cost=profieleren_dijkkern_cost,
            aanbregen_nieuwe_kleilaag_cost=aanbregen_nieuwe_kleilaag_cost,
            profieleren_vannieuwe_kleilaag_cost=profieleren_vannieuwe_kleilaag_cost,
            hergebruik_teelaarde_cost=hergebruik_teelaarde_cost,
            aanvullen_teelaarde_cost=aanvullen_teelaarde_cost,
            profieleren_nieuwe_graslaag_cost=profieleren_nieuwe_graslaag_cost
        )

    def calc_all_construction_costs(self, groundwork_cost: float, structure_cost: float) -> ConstructionCosts:
        totaal_benoemde_directe_bouwkosten = groundwork_cost + structure_cost
        if self.complexity == EnumerationComplexity.EASY:
            directe_bouwkosten = totaal_benoemde_directe_bouwkosten * (1 + self.surcharge_dict['Q-GGMAKNTD'].price_percent / 100)
        elif self.complexity == EnumerationComplexity.MEDIUM:
            directe_bouwkosten = totaal_benoemde_directe_bouwkosten * (1 + self.surcharge_dict['Q-GGGEMNTD'].price_percent / 100)
        elif self.complexity == EnumerationComplexity.HARD:
            directe_bouwkosten = totaal_benoemde_directe_bouwkosten * (1 + self.surcharge_dict['Q-GGMOENTD'].price_percent / 100)
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")


        pm_cost = directe_bouwkosten * self.surcharge_dict["Q-EKABKUKMAN"].price_percent / 100.0# Project management etc.
        general_cost = (directe_bouwkosten + pm_cost) * self.surcharge_dict["Q-AK"].price_percent / 100.0  # Algemene kosten
        risk_profit = (directe_bouwkosten + pm_cost + general_cost) * self.surcharge_dict["Q-WR"].price_percent / 100.0  # Winst & risico

        indirecte_bouwkosten = pm_cost + general_cost + risk_profit
        total_costs = directe_bouwkosten + indirecte_bouwkosten

        return ConstructionCosts(
            totale_BDBK_grondwerk=groundwork_cost,
            totale_BDBK_constructie=structure_cost,
            totale_benoemde_directe_bouwkosten=directe_bouwkosten,
            pm_kosten=pm_cost,
            algemene_kosten=general_cost,
            risico_en_winst=risk_profit,
            indirecte_bouwkosten=indirecte_bouwkosten,
            totale_bouwkosten=total_costs,
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


        general_cost = (direct_engineering_cost) * self.surcharge_dict["Q-AK"].price_percent / 100.0  # Algemene kosten
        risk_profit = (direct_engineering_cost + general_cost) * self.surcharge_dict["Q-WR"].price_percent / 100.0  # Winst & risico

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
        """

        :param construction_cost: Total construction cost from calc_all_construction_costs
        """

        insurances = construction_cost * self.surcharge_dict['Q-VERG'].price_percent / 100.0
        cables_pipes = construction_cost * self.surcharge_dict['Q-KL'].price_percent / 100.0
        damages = construction_cost * self.surcharge_dict['Q-PLAN'].price_percent / 100.0
        direct_general_costs = insurances + cables_pipes + damages

        genral_cost = direct_general_costs * self.surcharge_dict["Q-AK"].price_percent / 100.0
        risk_profit = (direct_general_costs + genral_cost) * self.surcharge_dict["Q-WR"].price_percent / 100.0

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
        """

        :param investering_cost: Sum of the construction total cost, engineering total cost and general total costs
        """
        if self.complexity == EnumerationComplexity.EASY:
            return investering_cost * self.surcharge_dict['Q-GGMAKONV'].price_percent / 100.0
        elif self.complexity == EnumerationComplexity.MEDIUM:
            return investering_cost * self.surcharge_dict['Q-GGGEMONV'].price_percent / 100.0
        elif self.complexity == EnumerationComplexity.HARD:
            return investering_cost * self.surcharge_dict['Q-GGMOEONV'].price_percent / 100.0
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")

    # def calc_direct_cost_structure(self, structure_model: StructureModel):
    def calc_direct_cost_structure(self, vaklengte: float, wandlengte: float, cost_function_parameters: dict) -> StructureCosts:
        c = cost_function_parameters['c']
        d = cost_function_parameters['d']
        z = cost_function_parameters['z']
        totale_directe_bouwkosten_per_meter = c *  wandlengte ** 2 + d * wandlengte + z

        return StructureCosts(
            totale_BDBK_constructie = totale_directe_bouwkosten_per_meter * vaklengte)