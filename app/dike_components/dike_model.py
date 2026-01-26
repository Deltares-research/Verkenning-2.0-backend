import geopandas as gpd

from app.cost_calculator import CostCalculator, DirectCostGroundWork, StructureCosts
from app.dike_components.ground_model import GroundModel
from app.dike_components.onverankerde_damwand_model import OnverankerdeDamwandModel
from app.dike_components.structure_model import StructureModel
from app.unit_costs_and_surcharges import load_kosten_catalogus
from pathlib import Path


class DikeModel:

    '''Model representing a dike with associated ground and structure models.'''
    def __init__(self, _3d_ground_polygon: gpd.GeoDataFrame = None, _2d_structure: gpd.GeoDataFrame = None, grid_size: float = 0.525, complexity: str = 'gemiddeld'):
        self.grid_size = grid_size  # Grid size for area calculations (default 0.525m for ~4070m² match)
        if _3d_ground_polygon is not None:
            self._3d_ground_polygon = _3d_ground_polygon
            self.ground_model = GroundModel(_3d_ground_polygon, grid_size=grid_size)
        if _2d_structure is not None:
            self._2d_structure = _2d_structure
            self._type = _2d_structure.loc[0,'type']
            if self._type == 'Onverankerde damwand':
                self.structure_model = OnverankerdeDamwandModel(_2d_structure, complexity=complexity)

    def compute_cost(self, nb_houses: int, road_area: float, complexity: str) -> dict:
        #set groundwork_cost and structure_costs to 0
        groundwork_cost = DirectCostGroundWork.zero()
        structure_costs = StructureCosts.zero()
        # of there is a ground model, compute groundwork cost
        #initialize CostCalculator here
        path_cost = Path("app/datasets/eenheidsprijzen.json")
        path_opslag_factor = Path("app/datasets/opslagfactoren.json")
        cat = load_kosten_catalogus(eenheidsprijzen=str(path_cost), opslagfactoren=str(path_opslag_factor))

        calculator = CostCalculator(cat, complexity)

        if hasattr(self, 'ground_model'):
            volumes = self.ground_model.calculate_all_dike_volumes()
            groundwork_cost = calculator.calc_direct_cost_ground_work(volumes=volumes)
        if hasattr(self, 'structure_model'):

            structure_costs = calculator.calc_direct_cost_structure(wandlengte=self.structure_model.wandlengte,
                                                                    vaklengte=self.structure_model.length,
                                                                    cost_function_parameters=self.structure_model.cost_function_parameters)
        


        total_direct_construction_cost = calculator.calc_all_construction_costs(groundwork_cost=groundwork_cost.groundwork_cost, structure_cost=structure_costs.directe_kosten_constructie) #add something
        engineering_cost = calculator.calc_all_engineering_costs(construction_cost=total_direct_construction_cost.total_costs)
        general_cost = calculator.calc_general_costs(construction_cost=total_direct_construction_cost.total_costs)
        investering_cost = total_direct_construction_cost.total_costs + engineering_cost.total_engineering_costs + general_cost.total_general_costs
        risk_cost = calculator.calc_risk_cost(investering_cost=investering_cost)
        real_estate_costs = calculator.calc_real_estate_costs(nb_houses=nb_houses, road_area=road_area)


        # total_cost_excl_BTW = investering_cost + risk_cost
        # print(total_cost_excl_BTW)


        return { 
            "Directe kosten grondwerk": groundwork_cost.to_dict(),
            "Directe kosten constructies": structure_costs.to_dict(),
            "Indirecte bouwkosten": total_direct_construction_cost.to_dict(),
            "Engineeringkosten": engineering_cost.to_dict(),
            "Overige bijkomende kosten": general_cost.to_dict(),
            "Risicoreservering": risk_cost,
            "Vastgoedkosten": real_estate_costs.to_dict(),
                }