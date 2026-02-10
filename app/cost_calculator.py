from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Self
from unicodedata import name

#dataclasses for catalogue items 
@dataclass
class SurchargeUnitItem:
    code: str
    price_percent: float  # renamed from prijs to make it clear it's a percentage
    description: str

@dataclass
class UnitPriceItem:
    code: str
    price: float  # renamed from prijs to make it clear it's a percentage
    description: str

#dataclasses for computed costs.
@dataclass
class SurchargeCostItem:
    code: str
    surcharge_percentage: float #original percentage from the catalog
    base_cost: float
    description: str = '' #Optional human readable name for the cost item, e.g. "Algemene kosten"

    @property
    def value(self) -> float:
        return self.base_cost * self.surcharge_percentage / 100.0
    
    @classmethod
    def zero(cls) -> "SurchargeCostItem":
        return SurchargeCostItem(code='', surcharge_percentage=0.0, base_cost=0.0)
    
    def __add__(self, other: Self) -> Self:
        #determine the new surcharge percentage based on the weights of the base costs
        total_base_cost = self.base_cost + other.base_cost
        if total_base_cost > 0:
            new_surcharge_percentage = (self.value + other.value) / total_base_cost * 100.0
        else:
            new_surcharge_percentage = 0.0
        return SurchargeCostItem(
            code='',
            surcharge_percentage=new_surcharge_percentage,
            base_cost=total_base_cost
        )

@dataclass
class CostItem:
    unit_cost: float
    quantity: float
    unit: str
    dimension: str = None #Optional for e.g. sheetpile length
    description: str = '' #Optional human readable description of the cost item, e.g. "Kosten voor het aanbrengen van een nieuwe kleilaag"

    @property
    def value(self) -> float:
        if self.quantity is not None:
            return self.unit_cost * self.quantity
        else:
            return 0

    def to_dict(self) -> dict:
        return {"value": self.value, "unit_cost": self.unit_cost, "quantity": self.quantity, 'unit': self.unit, "description": self.description}



@dataclass
class DirectCostGroundWork:
    kosten_opruimen: CostItem           
    kosten_maaien: CostItem             
    afgraven_toplaag: CostItem          
    afgraven_oud_materiaal: CostItem    
    hergebruik_oud_materiaal: CostItem  
    aanvullen_kern: CostItem            
    profileren_dijkkern: CostItem       
    aanbrengen_nieuwe_kleilaag: CostItem
    profileren_nieuwe_kleilaag: CostItem
    hergebruik_toplaag: CostItem
    aanvullen_toplaag: CostItem
    profileren_nieuwe_toplaag: CostItem
    inzaaien_nieuwe_toplaag: CostItem


    @property
    def totale_BDBK_grondwerk(self) -> float:
        """Benoemde Directe BouwKosten (BDBK)"""
        return (
            self.kosten_opruimen.value +
            self.kosten_maaien.value +
            self.afgraven_toplaag.value +
            self.afgraven_oud_materiaal.value +
            self.hergebruik_oud_materiaal.value +
            self.aanvullen_kern.value +
            self.profileren_dijkkern.value +
            self.aanbrengen_nieuwe_kleilaag.value +
            self.profileren_nieuwe_kleilaag.value +
            self.hergebruik_toplaag.value +
            self.aanvullen_toplaag.value +
            self.profileren_nieuwe_toplaag.value +
            self.inzaaien_nieuwe_toplaag.value
        )

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        data = {
            'kosten_opruimen': self.kosten_opruimen.to_dict(),
            'kosten_maaien': self.kosten_maaien.to_dict(),
            'afgraven_toplaag': self.afgraven_toplaag.to_dict(),
            'afgraven_oud_materiaal': self.afgraven_oud_materiaal.to_dict(),
            'hergebruik_oud_materiaal': self.hergebruik_oud_materiaal.to_dict(),
            'aanvullen_kern': self.aanvullen_kern.to_dict(),
            'profileren_dijkkern': self.profileren_dijkkern.to_dict(),
            'aanbrengen_nieuwe_kleilaag': self.aanbrengen_nieuwe_kleilaag.to_dict(),
            'profileren_nieuwe_kleilaag': self.profileren_nieuwe_kleilaag.to_dict(),
            'hergebruik_toplaag': self.hergebruik_toplaag.to_dict(),
            'aanvullen_toplaag': self.aanvullen_toplaag.to_dict(),
            'profileren_nieuwe_toplaag': self.profileren_nieuwe_toplaag.to_dict(),
            'inzaaien_nieuwe_toplaag': self.inzaaien_nieuwe_toplaag.to_dict(),
            'totale_BDBK_grondwerk': self.totale_BDBK_grondwerk
        }
        return data


    @classmethod
    def zero(cls) -> "DirectCostGroundWork":
        zero_cost = CostItem(unit_cost=0.0, quantity=0.0, unit='')
        return cls(**{field: zero_cost for field in cls.__dataclass_fields__})


@dataclass
class ConstructionCosts:
    #input values
    totale_BDBK_grondwerk: float  # Benoemde Directe BouwKosten (BDBK) (deel grond)
    totale_BDBK_constructie: float # Benoemde Directe BouwKosten (BDBK) (deel constructies)
    totale_BDBK_infrastructuur: float # Benoemde Directe BouwKosten (BDBK) (deel infrastructuur)

    #computed values
    directe_niet_benoemde_bouwkosten_grondwerk: SurchargeCostItem
    directe_niet_benoemde_bouwkosten_constructie: SurchargeCostItem
    directe_niet_benoemde_bouwkosten_infrastructuur: SurchargeCostItem
    pm_kosten: SurchargeCostItem
    algemene_kosten: SurchargeCostItem
    risico_en_winst: SurchargeCostItem

    @property
    def totale_directe_bouwkosten(self) -> float:
        return self.totale_BDBK_grondwerk + self.totale_BDBK_constructie + self.totale_BDBK_infrastructuur + self.directe_niet_benoemde_bouwkosten_grondwerk.value + self.directe_niet_benoemde_bouwkosten_constructie.value + self.directe_niet_benoemde_bouwkosten_infrastructuur.value
    @property
    def indirecte_bouwkosten(self) -> float:
        return self.pm_kosten.value + self.algemene_kosten.value + self.risico_en_winst.value
    @property
    def totale_bouwkosten(self) -> float:
        return self.totale_directe_bouwkosten + self.indirecte_bouwkosten
    @property
    def totale_directe_bouwkosten_grondwerk(self) -> float:
        return self.totale_BDBK_grondwerk + self.directe_niet_benoemde_bouwkosten_grondwerk.value
    @property
    def totale_directe_bouwkosten_constructie(self) -> float:
        return self.totale_BDBK_constructie + self.directe_niet_benoemde_bouwkosten_constructie.value
    @property
    def totale_directe_bouwkosten_infrastructuur(self) -> float:
        return self.totale_BDBK_infrastructuur + self.directe_niet_benoemde_bouwkosten_infrastructuur.value
    

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        return asdict(self)

    @classmethod
    def zero(cls) -> "ConstructionCosts":
        return cls(**{field: 0.0 for field in cls.__dataclass_fields__})
    
    #function to add two ConstructionCosts together
    def __add__(self, other: Self) -> Self:
        return ConstructionCosts(
            totale_BDBK_grondwerk=self.totale_BDBK_grondwerk + other.totale_BDBK_grondwerk,
            totale_BDBK_constructie=self.totale_BDBK_constructie + other.totale_BDBK_constructie,
            totale_BDBK_infrastructuur=self.totale_BDBK_infrastructuur + other.totale_BDBK_infrastructuur,
            totale_directe_bouwkosten=self.totale_directe_bouwkosten + other.totale_directe_bouwkosten,
            pm_kosten=self.pm_kosten + other.pm_kosten,
            algemene_kosten=self.algemene_kosten + other.algemene_kosten,
            risico_en_winst=self.risico_en_winst + other.risico_en_winst,
            indirecte_bouwkosten=self.indirecte_bouwkosten + other.indirecte_bouwkosten,
            totale_bouwkosten_grondwerk=self.totale_bouwkosten_grondwerk + other.totale_bouwkosten_grondwerk,
            totale_bouwkosten_constructie=self.totale_bouwkosten_constructie + other.totale_bouwkosten_constructie,
            totale_bouwkosten_infrastructuur=self.totale_bouwkosten_infrastructuur + other.totale_bouwkosten_infrastructuur,
            totale_bouwkosten=self.totale_bouwkosten + other.totale_bouwkosten
        )

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

    @classmethod
    def zero(cls) -> "EngineeringCosts":
        return cls(**{field: 0.0 for field in cls.__dataclass_fields__})
    
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
    direct_benoemd_real_estate_cost: CostItem
    direct_niet_benoemd_real_estate_cost: float
    indirect_real_estate_cost: float
    real_estate_risk_cost: float

    @property
    def total_real_estate_costs(self) -> float:
        return self.direct_benoemd_real_estate_cost.value + self.direct_niet_benoemd_real_estate_cost + self.indirect_real_estate_cost + self.real_estate_risk_cost

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
    def zero(cls) -> "StructureCosts":
        return cls(**{field: 0.0 for field in cls.__dataclass_fields__})

@dataclass
class InfrastructureCosts:
    remove_road: CostItem
    build_road: CostItem
    remove_bike_path: CostItem
    build_bike_path: CostItem

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        return asdict(self)

    @classmethod
    def zero(cls) -> "DirectCostGroundWork":
        return cls(**{field: 0.0 for field in cls.__dataclass_fields__})
    
    @property
    def totale_BDBK_infrastructuur(self) -> float:
        """Benoemde Directe BouwKosten (BDBK)"""
        return (
            self.remove_road.value +
            self.build_road.value +
            self.remove_bike_path.value +
            self.build_bike_path.value
        )
    
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
            'Percentages ter bepaling Opslagfactor investeringskosten / benoemde directe bouwkosten Constructief & Infra',
            'Percentages ter bepaling Opslagfactor investeringskosten / directe vastgoedkosten',
        ]

        categories_unit_prices = [
            'Grondverzet',
            'Profielafwerking',
            'Algemene werkzaamheden',
            'Wegen fietspaden en op-/afritten',
            'Vastgoed'
        ]

        all_items_surcharges = []
        for cat in categories_surcharges:
            all_items_surcharges.extend(catalogue.categorieen[cat])

        all_items_unit_prices = []
        for cat in categories_unit_prices:
            all_items_unit_prices.extend(catalogue.categorieen[cat])

        self.surcharge_dict: Dict[str, SurchargeUnitItem] = {
            item.code: SurchargeUnitItem(item.code, item.prijs, item.omschrijving)
            for item in all_items_surcharges
        }

        self.unit_price_dict: Dict[str, UnitPriceItem] = {
            item.code: UnitPriceItem(item.code, item.prijs, item.omschrijving)
            for item in all_items_unit_prices
        }

    def calc_real_estate_costs(self, nb_houses: float) -> RealEstateCosts:
        """
        Calculate real estate costs based on base cost and surcharge percentage.
        """
        direct_benoemd_real_estate_cost = CostItem(unit_cost=self.unit_price_dict['Q-VASTGOED'].price, quantity=nb_houses, unit='panden')
        if self.complexity == EnumerationComplexity.EASY:
            direct_niet_benoemd_real_estate_cost = direct_benoemd_real_estate_cost.value * self.surcharge_dict['Q-GVMAKNTD'].price_percent / 100.0
            _direct_real_estate_cost = direct_benoemd_real_estate_cost.value + direct_niet_benoemd_real_estate_cost
            indirect_real_estate_cost = _direct_real_estate_cost * self.surcharge_dict['Q-GVMAKIND'].price_percent / 100.0
            real_estate_risk_cost = (_direct_real_estate_cost+indirect_real_estate_cost) * self.surcharge_dict['Q-GVMAKNBO'].price_percent / 100.0
        elif self.complexity == EnumerationComplexity.MEDIUM:
            direct_niet_benoemd_real_estate_cost = direct_benoemd_real_estate_cost.value * self.surcharge_dict['Q-GVGEMNTD'].price_percent / 100.0
            _direct_real_estate_cost = direct_benoemd_real_estate_cost.value + direct_niet_benoemd_real_estate_cost
            indirect_real_estate_cost = _direct_real_estate_cost * self.surcharge_dict['Q-GVGEMIND'].price_percent / 100.0
            real_estate_risk_cost = (_direct_real_estate_cost+indirect_real_estate_cost) * self.surcharge_dict['Q-GVGEMNBO'].price_percent / 100.0
        elif self.complexity == EnumerationComplexity.HARD:
            direct_niet_benoemd_real_estate_cost = direct_benoemd_real_estate_cost.value * self.surcharge_dict['Q-GVMOENTD'].price_percent / 100.0
            _direct_real_estate_cost = direct_benoemd_real_estate_cost.value + direct_niet_benoemd_real_estate_cost
            indirect_real_estate_cost = _direct_real_estate_cost * self.surcharge_dict['Q-GVMOEIND'].price_percent / 100.0
            real_estate_risk_cost = (_direct_real_estate_cost+indirect_real_estate_cost) * self.surcharge_dict['Q-GVMOENBO'].price_percent / 100.0

        return RealEstateCosts(
            direct_benoemd_real_estate_cost=direct_benoemd_real_estate_cost,
            direct_niet_benoemd_real_estate_cost=direct_niet_benoemd_real_estate_cost,
            indirect_real_estate_cost=indirect_real_estate_cost,
            real_estate_risk_cost=real_estate_risk_cost,
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

        def build_cost_item(quantity, unit_cost_code, unit):
            return CostItem(quantity=quantity, unit_cost=self.unit_price_dict[unit_cost_code].price, unit=unit, description=self.unit_price_dict[unit_cost_code].description)
        
        ### Combine to get costs
        kosten_opruimen             = build_cost_item(S0, 'Q-GV010', 'm2') # opruimen terrein
        kosten_maaien               = build_cost_item(S0, 'Q-AW020', 'm2')  # maaien terrein
        afgraven_toplaag            = build_cost_item(V1b, 'Q-GV010', 'm3')  # afgraven oude grasbekleding naar depot
        afgraven_oud_materiaal      = build_cost_item(V2b, 'Q-GV030', 'm3')  # afgraven oude kleilaag en zand naar depot #TODO CHECK!
        hergebruik_oud_materiaal    = build_cost_item(V2b, 'Q-GV050', 'm3')  # hergebruiken oude kleilaag en zand in nieuwe kern #TODO CHECK!
        aanvullen_kern              = build_cost_item((V5 + V1b), 'Q-GV090', 'm3')  # aanvullen nieuwe kern met nieuw materiaal
        profileren_dijkkern         = build_cost_item(S5, 'Q-GV100', 'm2')  # profileren van dijkkern
        aanbrengen_nieuwe_kleilaag  = build_cost_item(V4, 'Q-GV080', 'm3')  # aanbrengen nieuwe kleilaag
        profileren_nieuwe_kleilaag  = build_cost_item(S5, 'Q-GV110', 'm2')  # profileren nieuwe kleilaag
        hergebruik_toplaag          = build_cost_item(V1b, 'Q-GV060', 'm3')  # hergebruiken teelaarde in nieuwe toplaag
        aanvullen_toplaag           = build_cost_item((V3 - V1b), 'Q-GV070', 'm3')  # aanvullen teelaarde in nieuwe toplaag
        profileren_nieuwe_toplaag   = build_cost_item(S5, 'Q-GV120', 'm2')  # profileren nieuwe graslaag en inzaaien
        inzaaien_nieuwe_toplaag     = build_cost_item(S5, 'Q-AW030', 'm2')  # profileren nieuwe graslaag en inzaaien
        

        return DirectCostGroundWork(
            kosten_opruimen=kosten_opruimen,
            kosten_maaien=kosten_maaien,
            afgraven_toplaag=afgraven_toplaag,
            afgraven_oud_materiaal=afgraven_oud_materiaal,
            hergebruik_oud_materiaal=hergebruik_oud_materiaal,
            aanvullen_kern=aanvullen_kern,
            profileren_dijkkern=profileren_dijkkern,
            aanbrengen_nieuwe_kleilaag=aanbrengen_nieuwe_kleilaag,
            profileren_nieuwe_kleilaag=profileren_nieuwe_kleilaag,
            hergebruik_toplaag=hergebruik_toplaag,
            aanvullen_toplaag=aanvullen_toplaag,
            profileren_nieuwe_toplaag=profileren_nieuwe_toplaag,
            inzaaien_nieuwe_toplaag=inzaaien_nieuwe_toplaag
        )

    def calc_construction_costs(self, groundwork_cost: float = 0.0, structure_cost: float = 0.0, infrastructure_cost: float = 0.0) -> ConstructionCosts:
        if self.complexity == EnumerationComplexity.EASY:
            directe_niet_benoemde_bouwkosten_grondwerk = SurchargeCostItem(code ='Q-GGMAKNTD', surcharge_percentage=self.surcharge_dict['Q-GGMAKNTD'].price_percent, base_cost=groundwork_cost)
            directe_niet_benoemde_bouwkosten_constructie = SurchargeCostItem(code ='Q-GCMAKNTD', surcharge_percentage=self.surcharge_dict['Q-GCMAKNTD'].price_percent, base_cost=structure_cost)
            directe_niet_benoemde_bouwkosten_infrastructuur = SurchargeCostItem(code ='Q-GCMAKNTD', surcharge_percentage=self.surcharge_dict['Q-GCMAKNTD'].price_percent, base_cost=infrastructure_cost)
        elif self.complexity == EnumerationComplexity.MEDIUM:
            directe_niet_benoemde_bouwkosten_grondwerk = SurchargeCostItem(code ='Q-GGGEMNTD', surcharge_percentage=self.surcharge_dict['Q-GGGEMNTD'].price_percent, base_cost=groundwork_cost)
            directe_niet_benoemde_bouwkosten_constructie = SurchargeCostItem(code ='Q-GCGEMNTD', surcharge_percentage=self.surcharge_dict['Q-GCGEMNTD'].price_percent, base_cost=structure_cost)
            directe_niet_benoemde_bouwkosten_infrastructuur = SurchargeCostItem(code ='Q-GCGEMNTD', surcharge_percentage=self.surcharge_dict['Q-GCGEMNTD'].price_percent, base_cost=infrastructure_cost)
        elif self.complexity == EnumerationComplexity.HARD:
            directe_niet_benoemde_bouwkosten_grondwerk = SurchargeCostItem(code ='Q-GGMOENTD', surcharge_percentage=self.surcharge_dict['Q-GGMOENTD'].price_percent, base_cost=groundwork_cost)
            directe_niet_benoemde_bouwkosten_constructie = SurchargeCostItem(code ='Q-GCMOENTD', surcharge_percentage=self.surcharge_dict['Q-GCMOENTD'].price_percent, base_cost=structure_cost)
            directe_niet_benoemde_bouwkosten_infrastructuur = SurchargeCostItem(code ='Q-GCMOENTD', surcharge_percentage=self.surcharge_dict['Q-GCMOENTD'].price_percent, base_cost=infrastructure_cost)
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")

        directe_niet_benoemde_bouwkosten = directe_niet_benoemde_bouwkosten_grondwerk.__add__(directe_niet_benoemde_bouwkosten_constructie).__add__(directe_niet_benoemde_bouwkosten_infrastructuur)

        directe_bouwkosten = groundwork_cost + structure_cost + infrastructure_cost + directe_niet_benoemde_bouwkosten.value

        pm_cost = SurchargeCostItem(code="Q-EKABKUKMAN", surcharge_percentage=self.surcharge_dict["Q-EKABKUKMAN"].price_percent, base_cost=directe_bouwkosten)
        general_cost = SurchargeCostItem(code="Q-AK", surcharge_percentage=self.surcharge_dict["Q-AK"].price_percent, base_cost=directe_bouwkosten + pm_cost.value)  # Algemene kosten
        risk_profit = SurchargeCostItem(code="Q-WR", surcharge_percentage=self.surcharge_dict["Q-WR"].price_percent, base_cost=directe_bouwkosten + pm_cost.value + general_cost.value)  # Winst & risico

        indirecte_bouwkosten = pm_cost.value + general_cost.value + risk_profit.value
        total_costs = directe_bouwkosten + indirecte_bouwkosten

        return ConstructionCosts(
            totale_BDBK_grondwerk=groundwork_cost,
            totale_BDBK_constructie=structure_cost,
            totale_BDBK_infrastructuur=infrastructure_cost,
            directe_niet_benoemde_bouwkosten_grondwerk = directe_niet_benoemde_bouwkosten_grondwerk,
            directe_niet_benoemde_bouwkosten_constructie=directe_niet_benoemde_bouwkosten_constructie,
            directe_niet_benoemde_bouwkosten_infrastructuur=directe_niet_benoemde_bouwkosten_infrastructuur,
            pm_kosten=pm_cost,
            algemene_kosten=general_cost,
            risico_en_winst=risk_profit,
        )

    def calc_construction_costs_structure(self, structure_cost: float) -> ConstructionCosts:
        if self.complexity == EnumerationComplexity.EASY:
            directe_niet_benoemde_bouwkosten_constructie = SurchargeCostItem(code ='Q-GCMAKNTD', surcharge_percentage=self.surcharge_dict['Q-GCMAKNTD'].price_percent, base_cost=structure_cost)
        elif self.complexity == EnumerationComplexity.MEDIUM:
            directe_niet_benoemde_bouwkosten_constructie = SurchargeCostItem(code ='Q-GCGEMNTD', surcharge_percentage=self.surcharge_dict['Q-GCGEMNTD'].price_percent, base_cost=structure_cost)
        elif self.complexity == EnumerationComplexity.HARD:
            directe_niet_benoemde_bouwkosten_constructie = SurchargeCostItem(code ='Q-GCMOENTD', surcharge_percentage=self.surcharge_dict['Q-GCMOENTD'].price_percent, base_cost=structure_cost)
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")
        directe_bouwkosten_constructie = structure_cost + directe_niet_benoemde_bouwkosten_constructie.value

        pm_cost = SurchargeCostItem(code="Q-EKABKUKMAN", surcharge_percentage=self.surcharge_dict["Q-EKABKUKMAN"].price_percent, base_cost=directe_bouwkosten_constructie)
        general_cost = SurchargeCostItem(code="Q-AK", surcharge_percentage=self.surcharge_dict["Q-AK"].price_percent, base_cost=directe_bouwkosten_constructie + pm_cost.value)  # Algemene kosten
        risk_profit = SurchargeCostItem(code="Q-WR", surcharge_percentage=self.surcharge_dict["Q-WR"].price_percent, base_cost=directe_bouwkosten_constructie + pm_cost.value + general_cost.value)  # Winst & risico

        indirecte_bouwkosten = pm_cost.value + general_cost.value + risk_profit.value
        total_costs = directe_bouwkosten_constructie + indirecte_bouwkosten

        return ConstructionCosts(
            totale_BDBK_grondwerk=0.0,
            totale_BDBK_constructie=structure_cost,
            totale_BDBK_infrastructuur=0.0,
            directe_niet_benoemde_bouwkosten_grondwerk = SurchargeCostItem.zero(),
            directe_niet_benoemde_bouwkosten_constructie=directe_niet_benoemde_bouwkosten_constructie,
            directe_niet_benoemde_bouwkosten_infrastructuur=SurchargeCostItem.zero(),
            pm_kosten=pm_cost,
            algemene_kosten=general_cost,
            risico_en_winst=risk_profit,
        )

    def calc_construction_costs_groundwork(self, groundwork_cost: float) -> ConstructionCosts:
        if self.complexity == EnumerationComplexity.EASY:
            directe_bouwkosten_grond = groundwork_cost * (1 + self.surcharge_dict['Q-GGMAKNTD'].price_percent / 100)
        elif self.complexity == EnumerationComplexity.MEDIUM:
            directe_bouwkosten_grond = groundwork_cost * (1 + self.surcharge_dict['Q-GGGEMNTD'].price_percent / 100)
        elif self.complexity == EnumerationComplexity.HARD:
            directe_bouwkosten_grond = groundwork_cost * (1 + self.surcharge_dict['Q-GGMOENTD'].price_percent / 100)
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")

        pm_cost = directe_bouwkosten_grond * self.surcharge_dict["Q-EKABKUKMAN"].price_percent / 100.0# Project management etc.
        general_cost = (directe_bouwkosten_grond + pm_cost) * self.surcharge_dict["Q-AK"].price_percent / 100.0  # Algemene kosten
        risk_profit = (directe_bouwkosten_grond + pm_cost + general_cost) * self.surcharge_dict["Q-WR"].price_percent / 100.0  # Winst & risico

        indirecte_bouwkosten = pm_cost + general_cost + risk_profit
        total_costs = directe_bouwkosten_grond + indirecte_bouwkosten

        return ConstructionCosts(
            totale_BDBK_grondwerk=groundwork_cost,
            totale_BDBK_infrastructuur=0.0,
            totale_BDBK_constructie=0.0,
            totale_directe_bouwkosten=directe_bouwkosten_grond,
            pm_kosten=pm_cost,
            algemene_kosten=general_cost,
            risico_en_winst=risk_profit,
            indirecte_bouwkosten=indirecte_bouwkosten,
            totale_bouwkosten_grondwerk=total_costs,
            totale_bouwkosten_constructie=0.0,
            totale_bouwkosten_infrastructuur=0.0,
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

    def calc_risk_cost(self, investering_cost: float, construction_costs: ConstructionCosts) -> float:
        """

        :param investering_cost: Sum of the construction total cost, engineering total cost and general total costs
        """
        #get the total investments for each component based on the share of the directe bouwkosten
        investering_grond = investering_cost * (construction_costs.totale_directe_bouwkosten_grondwerk/construction_costs.totale_directe_bouwkosten)
        investering_constructie = investering_cost * (construction_costs.totale_directe_bouwkosten_constructie/construction_costs.totale_directe_bouwkosten)
        investering_infra = investering_cost * (construction_costs.totale_directe_bouwkosten_infrastructuur/construction_costs.totale_directe_bouwkosten)
        if self.complexity == EnumerationComplexity.EASY:
            return (investering_grond * self.surcharge_dict['Q-GGMAKONV'].price_percent / 100.0) + (investering_constructie * self.surcharge_dict['Q-GCMAKONV'].price_percent / 100.0) + (investering_infra * self.surcharge_dict['Q-GCMAKONV'].price_percent / 100.0)
        elif self.complexity == EnumerationComplexity.MEDIUM:
            return (investering_grond * self.surcharge_dict['Q-GGGEMONV'].price_percent / 100.0) + (investering_constructie * self.surcharge_dict['Q-GCGEMONV'].price_percent / 100.0) + (investering_infra * self.surcharge_dict['Q-GCGEMONV'].price_percent / 100.0)
        elif self.complexity == EnumerationComplexity.HARD:
            return (investering_grond * self.surcharge_dict['Q-GGMOEONV'].price_percent / 100.0) + (investering_constructie * self.surcharge_dict['Q-GCMOEONV'].price_percent / 100.0) + (investering_infra * self.surcharge_dict['Q-GCMOEONV'].price_percent / 100.0)
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")

    # def calc_direct_cost_structure(self, structure_model: StructureModel):
    def calc_direct_cost_structure(self, vaklengte: float, wandlengte: float, cost_function_parameters: dict) -> StructureCosts:
        c = cost_function_parameters['c']
        d = cost_function_parameters['d']
        z = cost_function_parameters['z']
        totale_directe_bouwkosten_per_meter = c *  wandlengte ** 2 + d * wandlengte + z

        return StructureCosts(
            totale_BDBK_constructie = CostItem(unit_cost=totale_directe_bouwkosten_per_meter, quantity=vaklengte, unit='m', dimension=f'{wandlengte} m')
        )
    

    def calc_direct_cost_infrastructure(self, road_area: float, bike_path_area: float = 0) -> float:
        '''Assumption for now is that it is a regional road. We could improve this to distinguish between different types of roads. For now we set bike_path_area to 0 by default'''
        remove_road = CostItem(unit_cost=self.unit_price_dict['O-413'].price, quantity=road_area, unit = 'm2')  # removing regional road if there is one
        build_road = CostItem(unit_cost=self.unit_price_dict['O-513'].price, quantity=road_area, unit = 'm2')  # building regional road if there is one

        remove_bike_path = CostItem(unit_cost=self.unit_price_dict['O-410'].price, quantity=bike_path_area, unit = 'm2')  # removing bike path if there is one
        build_bike_path = CostItem(unit_cost=self.unit_price_dict['O-510'].price, quantity=bike_path_area, unit = 'm2')  # building bike path if there is one

        return InfrastructureCosts(remove_road=remove_road, build_road=build_road, remove_bike_path=remove_bike_path, build_bike_path=build_bike_path)

    def calc_construction_costs_infrastructure(self, infrastructure_cost: float) -> ConstructionCosts:
        '''Uses same surhcarges as structures'''
        if self.complexity == EnumerationComplexity.EASY:
            directe_bouwkosten_infrastructuur = infrastructure_cost * (1 + self.surcharge_dict['Q-GCMAKNTD'].price_percent / 100)
        elif self.complexity == EnumerationComplexity.MEDIUM:
            directe_bouwkosten_infrastructuur = infrastructure_cost * (1 + self.surcharge_dict['Q-GCGEMNTD'].price_percent / 100)
        elif self.complexity == EnumerationComplexity.HARD:
            directe_bouwkosten_infrastructuur = infrastructure_cost * (1 + self.surcharge_dict['Q-GCMOENTD'].price_percent / 100)
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")

        pm_cost = directe_bouwkosten_infrastructuur * self.surcharge_dict["Q-EKABKUKMAN"].price_percent / 100.0# Project management etc.
        general_cost = (directe_bouwkosten_infrastructuur + pm_cost) * self.surcharge_dict["Q-AK"].price_percent / 100.0  # Algemene kosten
        risk_profit = (directe_bouwkosten_infrastructuur + pm_cost + general_cost) * self.surcharge_dict["Q-WR"].price_percent / 100.0  # Winst & risico

        indirecte_bouwkosten = pm_cost + general_cost + risk_profit
        total_costs = directe_bouwkosten_infrastructuur + indirecte_bouwkosten

        return ConstructionCosts(
            totale_BDBK_grondwerk=0.0,
            totale_BDBK_constructie=0.0,
            totale_BDBK_infrastructuur=infrastructure_cost,
            totale_directe_bouwkosten=directe_bouwkosten_infrastructuur,
            pm_kosten=pm_cost,
            algemene_kosten=general_cost,
            risico_en_winst=risk_profit,
            indirecte_bouwkosten=indirecte_bouwkosten,
            totale_bouwkosten_grondwerk=0.0,
            totale_bouwkosten_constructie=0.0,
            totale_bouwkosten_infrastructuur=infrastructure_cost,
            totale_bouwkosten=total_costs,
        )