from app.dike_components.structure_model import StructureModel
from app.unit_costs_and_surcharges import load_kosten_catalogus, get_price

class HeaveschermModel(StructureModel):
    """Model representing an onverankerde damwand structure."""

    def set_cost_function_parameters(self) -> float:
        """Compute the total cost of the onverankerde damwand based on its properties."""
        self.cost_function_parameters = {'c': get_price(self.cost_catalog, 'c_Heavescherm'),
                                'd': get_price(self.cost_catalog, 'd_Heavescherm'),
                                'z': get_price(self.cost_catalog, 'z_Heavescherm')}