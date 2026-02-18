import geopandas as gpd

from app.cost_calculator import CostCalculator, DirectCostGroundWork, StructureCosts
from app.dike_components.ground_model import GroundModel
from app.dike_components.onverankerde_damwand_model import OnverankerdeDamwandModel
from app.dike_components.heavescherm_model import HeaveschermModel
from app.dike_components.verankerde_damwand_model import VerankerdeDamwandModel
from app.dike_components.structure_model import StructureModel
from app.unit_costs_and_surcharges import load_kosten_catalogus
from pathlib import Path


class DikeModel:
    '''Model representing a dike with associated ground and structure models.'''

    def __init__(self, _3d_ground_polygon: gpd.GeoDataFrame = None, _2d_structure: gpd.GeoDataFrame = None,
                 grid_size: float = 0.525, complexity: str = 'gemiddelde maatregel', excavation_mode: str = 'envelope'):
        self.grid_size = grid_size  # Grid size for area calculations (default 0.525m for ~4070m² match)
        self.complexity = complexity
        if _3d_ground_polygon is not None:
            self._3d_ground_polygon = _3d_ground_polygon
            self.ground_model = GroundModel(_3d_ground_polygon, grid_size=grid_size, excavation_mode=excavation_mode)
        if _2d_structure is not None:
            self._2d_structure = _2d_structure
            self._type = _2d_structure.loc[0, 'type']
            if self._type == 'Onverankerde damwand':
                self.structure_model = OnverankerdeDamwandModel(_2d_structure, complexity=complexity)
            elif self._type == 'Verankerde damwand':
                self.structure_model = VerankerdeDamwandModel(_2d_structure, complexity=complexity)
            elif self._type == 'Heavescherm':
                self.structure_model = HeaveschermModel(_2d_structure, complexity=complexity)
            else:
                raise ValueError(f"Onbekend constructietype: {self._type}")

    def compute_cost(self, nb_houses: int, road_area: float) -> dict:

        # set groundwork_cost and structure_costs to 0
        groundwork_cost = DirectCostGroundWork.zero()
        structure_costs = StructureCosts.zero()
        # of there is a ground model, compute groundwork cost
        # initialize CostCalculator here
        path_cost = Path(__file__).parent.parent.joinpath("datasets/eenheidsprijzen.json")
        path_opslag_factor = Path(__file__).parent.parent.joinpath("datasets/opslagfactoren.json")
        cat = load_kosten_catalogus(eenheidsprijzen=str(path_cost), opslagfactoren=str(path_opslag_factor))

        calculator = CostCalculator(cat, self.complexity)

        if hasattr(self, 'ground_model'):
            volumes = self.ground_model.calculate_all_dike_volumes()
            groundwork_cost = calculator.calc_direct_cost_ground_work(volumes=volumes)
        if hasattr(self, 'structure_model'):
            structure_costs = calculator.calc_direct_cost_structure(wandlengte=self.structure_model.wandlengte,
                                                                    vaklengte=self.structure_model.length,
                                                                    cost_function_parameters=self.structure_model.cost_function_parameters,
                                                                    structure_type=self.structure_model.constructietype)
        infrastructure_cost = calculator.calc_direct_cost_infrastructure(road_area=road_area)

        total_construction_cost = calculator.calc_construction_costs(
            groundwork_cost=groundwork_cost.totale_BDBK_grondwerk,
            structure_cost=structure_costs.totale_BDBK_constructie,
            infrastructure_cost=infrastructure_cost.totale_BDBK_infrastructuur)

        engineering_cost = calculator.calc_all_engineering_costs(
            construction_cost=total_construction_cost.totale_bouwkosten)

        general_cost = calculator.calc_general_costs(construction_cost=total_construction_cost.totale_bouwkosten)

        _investering_cost = total_construction_cost.totale_bouwkosten + engineering_cost.total_engineering_costs + general_cost.total_general_costs

        risk_cost = calculator.calc_risk_cost(investering_cost=_investering_cost,
                                              construction_costs=total_construction_cost)
        real_estate_costs = calculator.calc_real_estate_costs(nb_houses=nb_houses)

        # total_cost_excl_BTW = investering_cost + risk_cost
        # print(total_cost_excl_BTW)
        return {"Bouwkosten":
                    {"Directe Bouwkosten": {
                        "Directe kosten grondwerk": groundwork_cost.to_dict(),
                        "Directe kosten constructies": structure_costs.to_dict(),
                        "Directe kosten infrastructuur": infrastructure_cost.to_dict(),},
                    "Indirecte Bouwkosten": total_construction_cost.to_dict()},
                "Engineeringkosten" : engineering_cost.to_dict(),  # TODO Ideally follows the same structure as for Bouwkosten
                "Overige bijkomende kosten": general_cost.to_dict(), # TODO Ideally follows the same structure as for Bouwkosten
                "Risicoreservering": risk_cost.to_dict(),
                "Vastgoedkosten": real_estate_costs.to_dict(),
        }

