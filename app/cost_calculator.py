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

@dataclass
class SummedCostItem:
    '''Item for summing unit prices and/or surcharges together with and without BTW'''
    description: str
    value_excl_BTW: float
    value_incl_BTW: float

    def __add__(self, other: Self) -> Self:
        return SummedCostItem(description=self.description, value_excl_BTW=self.value_excl_BTW + other.value_excl_BTW, value_incl_BTW=self.value_incl_BTW + other.value_incl_BTW)
    
    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        data = {
            'description': self.description,
            'value_excl_BTW': self.value_excl_BTW,
            'value_incl_BTW': self.value_incl_BTW
        }
        return data
    
#dataclasses for computed costs.
@dataclass
class SurchargeCostItem:
    code: str
    surcharge_percentage: float #original percentage from the catalog
    base_cost: float    #Base cost on which the surcharge is applied. This should be without BTW
    description: str = '' #Optional human readable name for the cost item, e.g. "Algemene kosten"
    BTW: bool = True #Whether the cost item is subject to VAT (BTW in Dutch).
    BTW_percentage: float = 21.0 #The percentage of VAT to apply if BTW is True. Default is 21%, which is the standard VAT rate in the Netherlands.

    @property
    def value(self) -> float:
        return self.base_cost * self.surcharge_percentage / 100.0
    
    @property
    def value_incl_BTW(self) -> float:
        if self.BTW:
            return self.value * (1 + self.BTW_percentage / 100.0)
        else:
            return self.value
    
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

        #recompute averaged BTW percentage
        BTW_1  = self.value_incl_BTW - self.value
        BTW_2 = other.value_incl_BTW - other.value
        if total_base_cost > 0:
            new_BTW_percentage = (BTW_1 + BTW_2) / (total_base_cost * new_surcharge_percentage / 100.0) * 100.0
        else:
            new_BTW_percentage = 0.0
        return SurchargeCostItem(
            code='',
            surcharge_percentage=new_surcharge_percentage,
            base_cost=total_base_cost,
            BTW_percentage=new_BTW_percentage,
        )
    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        data = {
            'code': self.code,
            'surcharge_percentage': self.surcharge_percentage,
            'base_cost': self.base_cost,
            'value': self.value,
            'description': self.description,
        }

        return data

@dataclass
class CostItem:
    unit_cost: float
    quantity: float
    unit: str
    dimension: str = None #Optional for e.g. sheetpile length
    description: str = '' #Optional human readable description of the cost item, e.g. "Kosten voor het aanbrengen van een nieuwe kleilaag"
    BTW: str = True #Whether the cost item is subject to VAT (BTW in Dutch). 
    BTW_percentage: float = 21.0 #The percentage of VAT to apply if BTW is True. Default is 21%, which is the standard VAT rate in the Netherlands.

    @property
    def value(self) -> float:
        if self.quantity is not None:
            return self.unit_cost * self.quantity
        else:
            return 0

    @property 
    def value_incl_BTW(self) -> float:
        if self.BTW:
            return self.value * (1 + self.BTW_percentage / 100.0)
        else:
            return self.value
        
    def to_dict(self) -> dict:
        return {"value": self.value, "value_incl_BTW": self.value_incl_BTW, "unit_cost": self.unit_cost, "quantity": self.quantity, 'unit': self.unit, "description": self.description, "dimension": self.dimension}



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
    def totale_BDBK_grondwerk(self) -> SummedCostItem:
        """Benoemde Directe BouwKosten (BDBK)"""
        total_excl_BTW = (
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
        total_incl_BTW = (
            self. kosten_opruimen.value_incl_BTW +
            self.kosten_maaien.value_incl_BTW +
            self.afgraven_toplaag.value_incl_BTW +
            self.afgraven_oud_materiaal.value_incl_BTW +
            self.hergebruik_oud_materiaal.value_incl_BTW +
            self.aanvullen_kern.value_incl_BTW +
            self.profileren_dijkkern.value_incl_BTW +
            self.aanbrengen_nieuwe_kleilaag.value_incl_BTW +
            self.profileren_nieuwe_kleilaag.value_incl_BTW +
            self.hergebruik_toplaag.value_incl_BTW +
            self.aanvullen_toplaag.value_incl_BTW +
            self.profileren_nieuwe_toplaag.value_incl_BTW +
            self.inzaaien_nieuwe_toplaag.value_incl_BTW
        )

        return SummedCostItem(description="Benoemde directe bouwkosten grondwerk", value_excl_BTW=total_excl_BTW, value_incl_BTW=total_incl_BTW)

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
            'totale_BDBK_grondwerk': self.totale_BDBK_grondwerk.to_dict()
        }
        return data


    @classmethod
    def zero(cls) -> "DirectCostGroundWork":
        zero_cost = CostItem(unit_cost=0.0, quantity=0.0, unit='')
        return cls(**{field: zero_cost for field in cls.__dataclass_fields__})


@dataclass
class ConstructionCosts:
    #input values
    totale_BDBK_grondwerk: SummedCostItem  # Benoemde Directe BouwKosten (BDBK) (deel grond)
    totale_BDBK_constructie: SummedCostItem # Benoemde Directe BouwKosten (BDBK) (deel constructies)
    totale_BDBK_infrastructuur: SummedCostItem # Benoemde Directe BouwKosten (BDBK) (deel infrastructuur)

    #computed values
    directe_niet_benoemde_bouwkosten_grondwerk: SurchargeCostItem
    directe_niet_benoemde_bouwkosten_constructie: SurchargeCostItem
    directe_niet_benoemde_bouwkosten_infrastructuur: SurchargeCostItem
    pm_kosten: SurchargeCostItem
    algemene_kosten: SurchargeCostItem
    risico_en_winst: SurchargeCostItem

    @property
    def totale_directe_bouwkosten(self) -> SummedCostItem:
        return SummedCostItem(description = 'Totale directe bouwkosten', value_excl_BTW=
                              self.totale_BDBK_grondwerk.value_excl_BTW + self.totale_BDBK_constructie.value_excl_BTW + self.totale_BDBK_infrastructuur.value_excl_BTW + self.directe_niet_benoemde_bouwkosten_grondwerk.value + self.directe_niet_benoemde_bouwkosten_constructie.value + self.directe_niet_benoemde_bouwkosten_infrastructuur.value, value_incl_BTW=
                              self.totale_BDBK_grondwerk.value_incl_BTW + self.totale_BDBK_constructie.value_incl_BTW + self.totale_BDBK_infrastructuur.value_incl_BTW + self.directe_niet_benoemde_bouwkosten_grondwerk.value_incl_BTW + self.directe_niet_benoemde_bouwkosten_constructie.value_incl_BTW + self.directe_niet_benoemde_bouwkosten_infrastructuur.value_incl_BTW)
    @property
    def indirecte_bouwkosten(self) -> SummedCostItem:
        return SummedCostItem(description='Indirecte bouwkosten', value_excl_BTW=self.pm_kosten.value + self.algemene_kosten.value + self.risico_en_winst.value, value_incl_BTW=self.pm_kosten.value_incl_BTW + self.algemene_kosten.value_incl_BTW + self.risico_en_winst.value_incl_BTW)
    @property
    def totale_bouwkosten(self) -> SummedCostItem:
        return SummedCostItem(description='Totale bouwkosten', value_excl_BTW=self.totale_directe_bouwkosten.value_excl_BTW + self.indirecte_bouwkosten.value_excl_BTW, value_incl_BTW=self.totale_directe_bouwkosten.value_incl_BTW + self.indirecte_bouwkosten.value_incl_BTW)
    @property
    def totale_directe_bouwkosten_grondwerk(self) -> SummedCostItem:
        return SummedCostItem(description='Totale directe bouwkosten grondwerk', value_excl_BTW=self.totale_BDBK_grondwerk.value_excl_BTW + self.directe_niet_benoemde_bouwkosten_grondwerk.value, value_incl_BTW=self.totale_BDBK_grondwerk.value_incl_BTW + self.directe_niet_benoemde_bouwkosten_grondwerk.value_incl_BTW)
    @property
    def totale_directe_bouwkosten_constructie(self) -> SummedCostItem:
        return SummedCostItem(description='Totale directe bouwkosten constructie', value_excl_BTW=self.totale_BDBK_constructie.value_excl_BTW + self.directe_niet_benoemde_bouwkosten_constructie.value, value_incl_BTW=self.totale_BDBK_constructie.value_incl_BTW + self.directe_niet_benoemde_bouwkosten_constructie.value_incl_BTW)
    @property
    def totale_directe_bouwkosten_infrastructuur(self) -> SummedCostItem:
        return SummedCostItem(description='Totale directe bouwkosten infrastructuur', value_excl_BTW=self.totale_BDBK_infrastructuur.value_excl_BTW + self.directe_niet_benoemde_bouwkosten_infrastructuur.value, value_incl_BTW=self.totale_BDBK_infrastructuur.value_incl_BTW + self.directe_niet_benoemde_bouwkosten_infrastructuur.value_incl_BTW)
    

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        data = {
            'totale_BDBK_grondwerk': self.totale_BDBK_grondwerk.to_dict(),
            'totale_BDBK_constructie': self.totale_BDBK_constructie.to_dict(),
            'totale_BDBK_infrastructuur': self.totale_BDBK_infrastructuur.to_dict(),
            'directe_niet_benoemde_bouwkosten_grondwerk': self.directe_niet_benoemde_bouwkosten_grondwerk.to_dict(),
            'directe_niet_benoemde_bouwkosten_constructie': self.directe_niet_benoemde_bouwkosten_constructie.to_dict(),
            'directe_niet_benoemde_bouwkosten_infrastructuur': self.directe_niet_benoemde_bouwkosten_infrastructuur.to_dict(),
            'pm_kosten': self.pm_kosten.to_dict(),
            'algemene_kosten': self.algemene_kosten.to_dict(),
            'risico_en_winst': self.risico_en_winst.to_dict(),
            'totale_directe_bouwkosten': self.totale_directe_bouwkosten.to_dict(),
            'indirecte_bouwkosten': self.indirecte_bouwkosten.to_dict(),
            'totale_bouwkosten': self.totale_bouwkosten.to_dict(),
            }
        return data

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
    engineering_opdrachtgever: SurchargeCostItem  # Engineeringskosten opdrachtgever (EPK)
    engineering_opdrachtnemer: SurchargeCostItem  # Engineeringskosten opdrachtnemer (schets-, voor-, definitief ontwerp, e.d.)
    onderzoekskosten: SurchargeCostItem  # Onderzoeken (archeologie, explosieven, LNC, e.d.))


    algemene_kosten: SurchargeCostItem  # Algemene kosten (projectmanagement, voorbereiding, e.d.)
    winst_en_risico: SurchargeCostItem  # Winst en risico (onvoorziene omstandigheden, faalkosten, e.d.)

    @property
    def direct_engineering_cost(self) -> SummedCostItem:
        return SummedCostItem(
            description="Totale directe engineeringkosten",
            value_excl_BTW=self.engineering_opdrachtgever.value + self.engineering_opdrachtnemer.value + self.onderzoekskosten.value,
            value_incl_BTW=self.engineering_opdrachtgever.value_incl_BTW + self.engineering_opdrachtnemer.value_incl_BTW + self.onderzoekskosten.value_incl_BTW)
          

    @property
    def indirect_engineering_cost(self) -> SummedCostItem:
        return SummedCostItem(
            description="Totale indirecte engineeringkosten",
            value_excl_BTW=self.algemene_kosten.value + self.winst_en_risico.value,
            value_incl_BTW=self.algemene_kosten.value_incl_BTW + self.winst_en_risico.value_incl_BTW
        )
    
    @property
    def total_engineering_costs(self) -> SummedCostItem:
        return SummedCostItem(
            description="Totale engineeringkosten",
            value_excl_BTW=self.direct_engineering_cost.value_excl_BTW + self.indirect_engineering_cost.value_excl_BTW,
            value_incl_BTW=self.direct_engineering_cost.value_incl_BTW + self.indirect_engineering_cost.value_incl_BTW
        )

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        data = {
            'engineering_opdrachtgever': self.engineering_opdrachtgever.to_dict(),
            'engineering_opdrachtnemer': self.engineering_opdrachtnemer.to_dict(),
            'onderzoekskosten': self.onderzoekskosten.to_dict(),
            'algemene_kosten': self.algemene_kosten.to_dict(),
            'winst_en_risico': self.winst_en_risico.to_dict(),
            'direct_engineering_cost': self.direct_engineering_cost.to_dict(),
            'indirect_engineering_cost': self.indirect_engineering_cost.to_dict(),
            'total_engineering_costs': self.total_engineering_costs.to_dict()
        }
        return data

    @classmethod
    def zero(cls) -> "EngineeringCosts":
        return cls(**{field: 0.0 for field in cls.__dataclass_fields__})
    
@dataclass
class GeneralCosts:
    vergunningen_verzekeringen: SurchargeCostItem # Vergunningen, heffingen en verzekeringen
    kabels_leidingen: SurchargeCostItem # Kabels & leidingen
    planschade_inpassingsmaatregelen: SurchargeCostItem # Planschade & inpassingsmaatregelen
    algemene_kosten: SurchargeCostItem # Algemene kosten (projectmanagement, voorbereiding, e.d.)
    risico_en_winst: SurchargeCostItem # Winst en risico (onvoorziene omstandigheden, faalkosten, e.d.)

    @property
    def direct_general_costs(self) -> SummedCostItem:
        return SummedCostItem(
            description="Totale directe algemene kosten",
            value_excl_BTW=self.vergunningen_verzekeringen.value + self.kabels_leidingen.value + self.planschade_inpassingsmaatregelen.value,
            value_incl_BTW=self.vergunningen_verzekeringen.value_incl_BTW + self.kabels_leidingen.value_incl_BTW + self.planschade_inpassingsmaatregelen.value_incl_BTW
        )

    @property
    def indirect_general_costs(self) -> SummedCostItem:
        return SummedCostItem(
            description="Totale indirecte algemene kosten",
            value_excl_BTW=self.algemene_kosten.value + self.risico_en_winst.value,
            value_incl_BTW=self.algemene_kosten.value_incl_BTW + self.risico_en_winst.value_incl_BTW
        )
    @property
    def total_general_costs(self) -> SummedCostItem:
        return SummedCostItem(
            description="Totale algemene kosten",
            value_excl_BTW=self.direct_general_costs.value_excl_BTW + self.indirect_general_costs.value_excl_BTW,
            value_incl_BTW=self.direct_general_costs.value_incl_BTW + self.indirect_general_costs.value_incl_BTW
        )

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        data = {
            'vergunningen_verzekeringen': self.vergunningen_verzekeringen.to_dict(),
            'kabels_leidingen': self.kabels_leidingen.to_dict(),
            'planschade_inpassingsmaatregelen': self.planschade_inpassingsmaatregelen.to_dict(),
            'algemene_kosten': self.algemene_kosten.to_dict(),
            'risico_en_winst': self.risico_en_winst.to_dict(),
            'direct_general_costs': self.direct_general_costs.to_dict(),
            'indirect_general_costs': self.indirect_general_costs.to_dict(),
            'total_general_costs': self.total_general_costs.to_dict()
        }
        return data

@dataclass
class RealEstateCosts:
    direct_benoemd_real_estate_cost: CostItem
    direct_niet_benoemd_real_estate_cost: SurchargeCostItem
    indirect_real_estate_cost: SurchargeCostItem
    real_estate_risk_cost: SurchargeCostItem

    @property
    def total_real_estate_costs(self) -> float:
        return SummedCostItem(
            description="Totale vastgoedkosten",
            value_excl_BTW=self.direct_benoemd_real_estate_cost.value + self.direct_niet_benoemd_real_estate_cost.value + self.indirect_real_estate_cost.value + self.real_estate_risk_cost.value,
            value_incl_BTW=self.direct_benoemd_real_estate_cost.value_incl_BTW + self.direct_niet_benoemd_real_estate_cost.value_incl_BTW + self.indirect_real_estate_cost.value_incl_BTW + self.real_estate_risk_cost.value_incl_BTW
        )

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        data = {
            'direct_benoemd_real_estate_cost': self.direct_benoemd_real_estate_cost.to_dict(),
            'direct_niet_benoemd_real_estate_cost': self.direct_niet_benoemd_real_estate_cost.to_dict(),
            'indirect_real_estate_cost': self.indirect_real_estate_cost.to_dict(),
            'real_estate_risk_cost': self.real_estate_risk_cost.to_dict(),
            'total_real_estate_costs': self.total_real_estate_costs.to_dict()
        }
        return data


@dataclass
class StructureCosts:
    directe_bouwkosten: CostItem

    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        data = {
            'directe_bouwkosten': self.directe_bouwkosten.to_dict(),
            'totale_BDBK_constructie': self.totale_BDBK_constructie.to_dict()
        }
        return data

    @property
    def totale_BDBK_constructie(self) -> SummedCostItem:
        """Benoemde Directe BouwKosten (BDBK)"""
        return SummedCostItem(description="Benoemde directe bouwkosten constructie", value_excl_BTW=self.directe_bouwkosten.value, value_incl_BTW=self.directe_bouwkosten.value_incl_BTW)
    
    @classmethod
    def zero(cls) -> "StructureCosts":
        zero_cost = CostItem(unit_cost=0.0, quantity=0.0, unit='')
        return cls(**{field: zero_cost for field in cls.__dataclass_fields__})

@dataclass
class InfrastructureCosts:
    verwijderen_weg: CostItem
    aanleggen_weg: CostItem
    verwijderen_fietspad: CostItem
    aanleggen_fietspad: CostItem

    @property
    def totale_BDBK_infrastructuur(self) -> SummedCostItem:
        """Benoemde Directe BouwKosten (BDBK)"""
        return SummedCostItem(
            description="Benoemde directe bouwkosten infrastructuur",
            value_excl_BTW=(
                self.verwijderen_weg.value +
                self.aanleggen_weg.value +
                self.verwijderen_fietspad.value +
                self.aanleggen_fietspad.value
            ),
            value_incl_BTW=(
                self.verwijderen_weg.value_incl_BTW +
                self.aanleggen_weg.value_incl_BTW +
                self.verwijderen_fietspad.value_incl_BTW +
                self.aanleggen_fietspad.value_incl_BTW
            )
        )
    
    def to_dict(self) -> dict:
        """Serialize the dataclass to a dict"""
        data = {
            'verwijderen_weg': self.verwijderen_weg.to_dict(),
            'aanleggen_weg': self.aanleggen_weg.to_dict(),
            'verwijderen_fietspad': self.verwijderen_fietspad.to_dict(),
            'aanleggen_fietspad': self.aanleggen_fietspad.to_dict(),
            'totale_BDBK_infrastructuur': self.totale_BDBK_infrastructuur.to_dict()
        }
        return data

    @classmethod
    def zero(cls) -> "InfrastructureCosts":
        zero_cost = CostItem(unit_cost=0.0, quantity=0.0, unit='')
        return cls(**{field: zero_cost for field in cls.__dataclass_fields__})
    
    
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

    def build_cost_item(self, quantity, unit_cost_code, unit):
        return CostItem(quantity=quantity, unit_cost=self.unit_price_dict[unit_cost_code].price, unit=unit, description=self.unit_price_dict[unit_cost_code].description)

    def build_surcharge_cost_item(self, base_cost, surcharge_code, BTW = True):
        surcharge_percentage = self.surcharge_dict[surcharge_code].price_percent
        description = self.surcharge_dict[surcharge_code].description
        return SurchargeCostItem(code=surcharge_code, surcharge_percentage=surcharge_percentage, base_cost=base_cost, description=description, BTW = BTW)
    
    def calc_real_estate_costs(self, nb_houses: float) -> RealEstateCosts:
        """
        Calculate real estate costs based on base cost and surcharge percentage.
        """
        direct_benoemd_real_estate_cost = CostItem(unit_cost=self.unit_price_dict['Q-VASTGOED'].price, quantity=nb_houses, unit='panden')
        if self.complexity == EnumerationComplexity.EASY:
            direct_niet_benoemd_real_estate_cost = self.build_surcharge_cost_item(base_cost=direct_benoemd_real_estate_cost.value, surcharge_code='Q-GVMAKNTD', BTW=False)
            _direct_real_estate_cost = direct_benoemd_real_estate_cost.value + direct_niet_benoemd_real_estate_cost.value
            indirect_real_estate_cost = self.build_surcharge_cost_item(base_cost=_direct_real_estate_cost, surcharge_code='Q-GVMAKIND', BTW=False)
            real_estate_risk_cost = self.build_surcharge_cost_item(base_cost=_direct_real_estate_cost+indirect_real_estate_cost.value, surcharge_code='Q-GVMAKNBO', BTW=False)
        elif self.complexity == EnumerationComplexity.MEDIUM:
            direct_niet_benoemd_real_estate_cost = self.build_surcharge_cost_item(base_cost=direct_benoemd_real_estate_cost.value, surcharge_code='Q-GVGEMNTD', BTW=False)
            _direct_real_estate_cost = direct_benoemd_real_estate_cost.value + direct_niet_benoemd_real_estate_cost.value
            indirect_real_estate_cost = self.build_surcharge_cost_item(base_cost=_direct_real_estate_cost, surcharge_code='Q-GVGEMIND', BTW=False)
            real_estate_risk_cost = self.build_surcharge_cost_item(base_cost=_direct_real_estate_cost+indirect_real_estate_cost.value, surcharge_code='Q-GVGEMNBO', BTW=False)
        elif self.complexity == EnumerationComplexity.HARD:
            direct_niet_benoemd_real_estate_cost = self.build_surcharge_cost_item(base_cost=direct_benoemd_real_estate_cost.value, surcharge_code='Q-GVMOENTD', BTW=False)
            _direct_real_estate_cost = direct_benoemd_real_estate_cost.value + direct_niet_benoemd_real_estate_cost.value
            indirect_real_estate_cost = self.build_surcharge_cost_item(base_cost=_direct_real_estate_cost, surcharge_code='Q-GVMOEIND', BTW=False)
            real_estate_risk_cost = self.build_surcharge_cost_item(base_cost=_direct_real_estate_cost+indirect_real_estate_cost.value, surcharge_code='Q-GVMOENBO', BTW=False)

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
        V1b = volumes['V1b']  # Volume grasbekleding van het huidig profiel (verwijderd en hergebruikt) - BASED ON envelop_AHN_surface
        V2b = volumes['V2b']  # Volume kleilaag van het huidig profiel (verwijderd en hergebruikt als kernmateriaal) - BASED ON envelop_AHN_surface
        V3 = volumes['V3']  # volume grasbekleding van de nieuwe dijk - BASED ON envelop_design_surface
        V4 = volumes['V4']  # volume kleilaag van de nieuwe dijk - BASED ON envelop_design_surface
        V5 = volumes['V5']  # volume kernmateriaal van de nieuwe dijk - BASED ON envelop_design_surface
        full_AHN_surface = volumes['full_AHN_surface'] # m2
        envelop_AHN_surface = volumes['envelop_AHN_surface'] # m2
        full_design_surface = volumes['full_design_surface'] # m2
        envelop_design_surface = volumes['envelop_design_surface'] # m2

        ### Combine to get costs
        kosten_opruimen             = self.build_cost_item(full_AHN_surface, 'Q-AW010', 'm2') # opruimen terrein
        kosten_maaien               = self.build_cost_item(full_AHN_surface, 'Q-AW020', 'm2')  # maaien terrein
        afgraven_toplaag            = self.build_cost_item(V1b, 'Q-GV010', 'm3')  # afgraven oude grasbekleding naar depot
        afgraven_oud_materiaal      = self.build_cost_item(V2b, 'Q-GV030', 'm3')  # afgraven oude kleilaag en zand naar depot #TODO CHECK!
        hergebruik_oud_materiaal    = self.build_cost_item(V2b, 'Q-GV050', 'm3')  # hergebruiken oude kleilaag en zand in nieuwe kern. Als opvulling volume oude kleilaag.
        aanvullen_kern              = self.build_cost_item(V5 + V1b, 'Q-GV090', 'm3')  # aanvullen nieuwe kern met nieuw materiaal. V2b is al aangevuld
        profileren_dijkkern         = self.build_cost_item(envelop_design_surface, 'Q-GV100', 'm2')  # profileren van dijkkern
        aanbrengen_nieuwe_kleilaag  = self.build_cost_item(V4, 'Q-GV080', 'm3')  # aanbrengen nieuwe kleilaag
        profileren_nieuwe_kleilaag  = self.build_cost_item(envelop_design_surface, 'Q-GV110', 'm2')  # profileren nieuwe kleilaag
        hergebruik_toplaag          = self.build_cost_item(V1b, 'Q-GV060', 'm3')  # hergebruiken teelaarde in nieuwe toplaag
        aanvullen_toplaag           = self.build_cost_item(max(0, V3 - V1b), 'Q-GV070', 'm3')  # aanvullen teelaarde in nieuwe toplaag
        profileren_nieuwe_toplaag   = self.build_cost_item(full_design_surface, 'Q-GV120', 'm2')  # profileren nieuwe graslaag en inzaaien
        inzaaien_nieuwe_toplaag     = self.build_cost_item(full_design_surface, 'Q-AW030', 'm2')  # profileren nieuwe graslaag en inzaaien
        

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

    def calc_construction_costs(self, 
                                groundwork_cost: SummedCostItem = SummedCostItem(description='Groundwork cost', value_excl_BTW=0.0, value_incl_BTW=0.0), 
                                structure_cost: SummedCostItem = SummedCostItem(description='Structure cost', value_excl_BTW=0.0, value_incl_BTW=0.0), 
                                infrastructure_cost: SummedCostItem = SummedCostItem(description='Infrastructure cost', value_excl_BTW=0.0, value_incl_BTW=0.0)) -> ConstructionCosts:
        '''Input costs are excl. BTW, as the surcharge percentages are applied on the base cost excl. BTW'''
        
        if self.complexity == EnumerationComplexity.EASY:
            directe_niet_benoemde_bouwkosten_grondwerk = SurchargeCostItem(code ='Q-GGMAKNTD', surcharge_percentage=self.surcharge_dict['Q-GGMAKNTD'].price_percent, base_cost=groundwork_cost.value_excl_BTW)
            directe_niet_benoemde_bouwkosten_constructie = SurchargeCostItem(code ='Q-GCMAKNTD', surcharge_percentage=self.surcharge_dict['Q-GCMAKNTD'].price_percent, base_cost=structure_cost.value_excl_BTW)
            directe_niet_benoemde_bouwkosten_infrastructuur = SurchargeCostItem(code ='Q-GCMAKNTD', surcharge_percentage=self.surcharge_dict['Q-GCMAKNTD'].price_percent, base_cost=infrastructure_cost.value_excl_BTW)
        elif self.complexity == EnumerationComplexity.MEDIUM:
            directe_niet_benoemde_bouwkosten_grondwerk = SurchargeCostItem(code ='Q-GGGEMNTD', surcharge_percentage=self.surcharge_dict['Q-GGGEMNTD'].price_percent, base_cost=groundwork_cost.value_excl_BTW)
            directe_niet_benoemde_bouwkosten_constructie = SurchargeCostItem(code ='Q-GCGEMNTD', surcharge_percentage=self.surcharge_dict['Q-GCGEMNTD'].price_percent, base_cost=structure_cost.value_excl_BTW)
            directe_niet_benoemde_bouwkosten_infrastructuur = SurchargeCostItem(code ='Q-GCGEMNTD', surcharge_percentage=self.surcharge_dict['Q-GCGEMNTD'].price_percent, base_cost=infrastructure_cost.value_excl_BTW)
        elif self.complexity == EnumerationComplexity.HARD:
            directe_niet_benoemde_bouwkosten_grondwerk = SurchargeCostItem(code ='Q-GGMOENTD', surcharge_percentage=self.surcharge_dict['Q-GGMOENTD'].price_percent, base_cost=groundwork_cost.value_excl_BTW)
            directe_niet_benoemde_bouwkosten_constructie = SurchargeCostItem(code ='Q-GCMOENTD', surcharge_percentage=self.surcharge_dict['Q-GCMOENTD'].price_percent, base_cost=structure_cost.value_excl_BTW)
            directe_niet_benoemde_bouwkosten_infrastructuur = SurchargeCostItem(code ='Q-GCMOENTD', surcharge_percentage=self.surcharge_dict['Q-GCMOENTD'].price_percent, base_cost=infrastructure_cost.value_excl_BTW)
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")

        directe_niet_benoemde_bouwkosten = directe_niet_benoemde_bouwkosten_grondwerk.__add__(directe_niet_benoemde_bouwkosten_constructie).__add__(directe_niet_benoemde_bouwkosten_infrastructuur)

        directe_bouwkosten = groundwork_cost.__add__(structure_cost).__add__(infrastructure_cost).__add__(SummedCostItem(description='Directe niet benoemde bouwkosten', value_excl_BTW=directe_niet_benoemde_bouwkosten.value, value_incl_BTW=directe_niet_benoemde_bouwkosten.value_incl_BTW))

        pm_cost = SurchargeCostItem(code="Q-EKABKUKMAN", surcharge_percentage=self.surcharge_dict["Q-EKABKUKMAN"].price_percent, base_cost=directe_bouwkosten.value_excl_BTW)
        general_cost = SurchargeCostItem(code="Q-AK", surcharge_percentage=self.surcharge_dict["Q-AK"].price_percent, base_cost=directe_bouwkosten.value_excl_BTW + pm_cost.value)  # Algemene kosten
        risk_profit = SurchargeCostItem(code="Q-WR", surcharge_percentage=self.surcharge_dict["Q-WR"].price_percent, base_cost=directe_bouwkosten.value_excl_BTW + pm_cost.value + general_cost.value)  # Winst & risico

        indirecte_bouwkosten = SummedCostItem(description='Indirecte bouwkosten', value_excl_BTW=pm_cost.value + general_cost.value + risk_profit.value, value_incl_BTW=pm_cost.value_incl_BTW + general_cost.value_incl_BTW + risk_profit.value_incl_BTW)
        total_costs = directe_bouwkosten.__add__(indirecte_bouwkosten)

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

    def calc_all_engineering_costs(self, construction_cost: SummedCostItem) -> EngineeringCosts:
        """

        :param construction_cost: Total construction cost from calc_all_construction_costs: SummedCostItem
        """

        if self.complexity == EnumerationComplexity.EASY:
            engineering_opdrachtgever = self.build_surcharge_cost_item(base_cost=construction_cost.value_excl_BTW, surcharge_code="Q-ENGOG1", BTW=False)
            engineering_opdrachtnemer = self.build_surcharge_cost_item(base_cost=construction_cost.value_excl_BTW, surcharge_code="Q-ENGON1")

        elif self.complexity == EnumerationComplexity.MEDIUM:
            engineering_opdrachtgever = self.build_surcharge_cost_item(base_cost=construction_cost.value_excl_BTW, surcharge_code="Q-ENGOG2", BTW=False)
            engineering_opdrachtnemer = self.build_surcharge_cost_item(base_cost=construction_cost.value_excl_BTW, surcharge_code="Q-ENGON2")

        elif self.complexity == EnumerationComplexity.HARD:
            engineering_opdrachtgever = self.build_surcharge_cost_item(base_cost=construction_cost.value_excl_BTW, surcharge_code="Q-ENGOG3", BTW=False)
            engineering_opdrachtnemer = self.build_surcharge_cost_item(base_cost=construction_cost.value_excl_BTW, surcharge_code="Q-ENGON3")
        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")
        #TODO Check with Peter!
        research_cost = self.build_surcharge_cost_item(base_cost=construction_cost.value_excl_BTW, surcharge_code="Q-OND")
        
        direct_engineering_cost_OG = engineering_opdrachtgever.value 
        direct_engineering_cost_others = engineering_opdrachtnemer.value + research_cost.value

        
        general_cost_part1 = self.build_surcharge_cost_item(base_cost=direct_engineering_cost_OG, surcharge_code="Q-AK", BTW=False)
        general_cost_part2 = self.build_surcharge_cost_item(base_cost=direct_engineering_cost_others, surcharge_code="Q-AK", BTW=True)
        general_cost = general_cost_part1.__add__(general_cost_part2)

        risk_profit_part1 = self.build_surcharge_cost_item(base_cost=direct_engineering_cost_OG + general_cost_part1.value, surcharge_code="Q-WR", BTW=False)
        risk_profit_part2 = self.build_surcharge_cost_item(base_cost=direct_engineering_cost_others + general_cost_part2.value, surcharge_code="Q-WR", BTW=True)
        risk_profit = risk_profit_part1.__add__(risk_profit_part2)


        return EngineeringCosts(
            engineering_opdrachtgever=engineering_opdrachtgever,
            engineering_opdrachtnemer=engineering_opdrachtnemer,
            onderzoekskosten=research_cost,
            algemene_kosten=general_cost,
            winst_en_risico=risk_profit,
        )

    def calc_general_costs(self, construction_cost: SummedCostItem) -> GeneralCosts:
        """

        :param construction_cost: Total construction cost from calc_all_construction_costs
        """
        vergunningen_verzekeringen = self.build_surcharge_cost_item(base_cost=construction_cost.value_excl_BTW, surcharge_code='Q-VERG')
        kabels_leidingen = self.build_surcharge_cost_item(base_cost=construction_cost.value_excl_BTW, surcharge_code='Q-KL')
        planschade_inpassingsmaatregelen = self.build_surcharge_cost_item(base_cost=construction_cost.value_excl_BTW, surcharge_code='Q-PLAN')
        
        
        direct_general_costs = vergunningen_verzekeringen.value + kabels_leidingen.value + planschade_inpassingsmaatregelen.value

        algemene_kosten = self.build_surcharge_cost_item(base_cost=direct_general_costs, surcharge_code="Q-AK")
        risico_winst = self.build_surcharge_cost_item(base_cost=direct_general_costs + algemene_kosten.value, surcharge_code="Q-WR")

        return GeneralCosts(
            vergunningen_verzekeringen=vergunningen_verzekeringen,
            kabels_leidingen=kabels_leidingen,
            planschade_inpassingsmaatregelen=planschade_inpassingsmaatregelen,
            algemene_kosten=algemene_kosten,
            risico_en_winst=risico_winst,
        )

    def calc_risk_cost(self, investering_cost: SummedCostItem, construction_costs: ConstructionCosts) -> float:
        """

        :param investering_cost: Sum of the construction total cost, engineering total cost and general total costs
        """
        #get the total investments for each component based on the share of the directe bouwkosten. We first do excl. BTW, as the surcharge percentages are applied on the base cost excl. BTW
        investering_grond = investering_cost.value_excl_BTW * (construction_costs.totale_directe_bouwkosten_grondwerk.value_excl_BTW/construction_costs.totale_directe_bouwkosten.value_excl_BTW) if construction_costs.totale_directe_bouwkosten.value_excl_BTW > 0 else 0.0
        investering_constructie = investering_cost.value_excl_BTW * (construction_costs.totale_directe_bouwkosten_constructie.value_excl_BTW/construction_costs.totale_directe_bouwkosten.value_excl_BTW) if construction_costs.totale_directe_bouwkosten.value_excl_BTW > 0 else 0.0
        investering_infra = investering_cost.value_excl_BTW * (construction_costs.totale_directe_bouwkosten_infrastructuur.value_excl_BTW/construction_costs.totale_directe_bouwkosten.value_excl_BTW) if construction_costs.totale_directe_bouwkosten.value_excl_BTW > 0 else 0.0
        if self.complexity == EnumerationComplexity.EASY:
            risk_grond = self.build_surcharge_cost_item(base_cost=investering_grond, surcharge_code='Q-GGMAKONV')
            risk_constructie = self.build_surcharge_cost_item(base_cost=investering_constructie, surcharge_code='Q-GCMAKONV')
            risk_infra = self.build_surcharge_cost_item(base_cost=investering_infra, surcharge_code='Q-GCMAKONV')
            risk_total = risk_grond.__add__(risk_constructie).__add__(risk_infra)
            risk_total.description = "Objectoverstijgende risicoreservering (makkelijk)"
            return risk_total
        elif self.complexity == EnumerationComplexity.MEDIUM:
            risk_grond = self.build_surcharge_cost_item(base_cost=investering_grond, surcharge_code='Q-GGGEMONV')
            risk_constructie = self.build_surcharge_cost_item(base_cost=investering_constructie, surcharge_code='Q-GCGEMONV')
            risk_infra = self.build_surcharge_cost_item(base_cost=investering_infra, surcharge_code='Q-GCGEMONV')
            risk_total = risk_grond.__add__(risk_constructie).__add__(risk_infra)
            risk_total.description = "Objectoverstijgende risicoreservering (gemiddeld)"
            return risk_total

        elif self.complexity == EnumerationComplexity.HARD:
            risk_grond = self.build_surcharge_cost_item(base_cost=investering_grond, surcharge_code='Q-GGMOEONV')
            risk_constructie = self.build_surcharge_cost_item(base_cost=investering_constructie, surcharge_code='Q-GCMOEONV')
            risk_infra = self.build_surcharge_cost_item(base_cost=investering_infra, surcharge_code='Q-GCMOEONV')
            risk_total = risk_grond.__add__(risk_constructie).__add__(risk_infra)
            risk_total.description = "Objectoverstijgende risicoreservering (moeilijk)"
            return risk_total

        else:
            raise ValueError(f"Unsupported complexity level: {self.complexity}")

    # def calc_direct_cost_structure(self, structure_model: StructureModel):
    def calc_direct_cost_structure(self, vaklengte: float, wandlengte: float, cost_function_parameters: dict, structure_type: str) -> StructureCosts:
        c = cost_function_parameters['c']
        d = cost_function_parameters['d']
        z = cost_function_parameters['z']
        totale_directe_bouwkosten_per_meter = c *  wandlengte ** 2 + d * wandlengte + z if wandlengte > 0 else 0.0
        directe_bouwkosten = CostItem(unit_cost=totale_directe_bouwkosten_per_meter, quantity=vaklengte, unit='m', description=structure_type,dimension=f'{wandlengte:.1f} m wandlengte')
        return StructureCosts(
            directe_bouwkosten = directe_bouwkosten,
        )
    

    def calc_direct_cost_infrastructure(self, road_area: float, bike_path_area: float = 0) -> float:
        '''Assumption for now is that it is a regional road. We could improve this to distinguish between different types of roads. For now we set bike_path_area to 0 by default'''
        verwijderen_weg = self.build_cost_item(quantity=road_area, unit_cost_code='O-413', unit='m2')  # removing regional road if there is one
        aanleggen_weg = self.build_cost_item(quantity=road_area, unit_cost_code='O-513', unit='m2')  # building regional road if there is one

        verwijderen_fietspad = self.build_cost_item(quantity=bike_path_area, unit_cost_code='O-410', unit='m2')  # removing bike path if there is one
        aanleggen_fietspad = self.build_cost_item(quantity=bike_path_area, unit_cost_code='O-510', unit='m2')  # building bike path if there is one

        return InfrastructureCosts(verwijderen_weg=verwijderen_weg, aanleggen_weg=aanleggen_weg, verwijderen_fietspad=verwijderen_fietspad, aanleggen_fietspad=aanleggen_fietspad)

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