from app.structures.structure_model import StructureModel
from app.unit_costs_and_surcharges import load_kosten_catalogus, get_price

class OnverankerdeDamwandModel(StructureModel):
    """Model representing an onverankerde damwand structure."""

    def compute_cost(self) -> float:
        """Compute the total cost of the onverankerde damwand based on its properties."""
        # Example cost calculation logic
        self.cost_components = {}
        #compute the direct building costs
        self.compute_directe_bouwkosten(c = get_price(self.cost_catalog, 'c_Onverankerd'),
                                                                     d = get_price(self.cost_catalog, 'd_Onverankerd'),
                                                                    z = get_price(self.cost_catalog, 'z_Onverankerd'))
        
        

    def compute_directe_bouwkosten(self, c, d, z) -> float:
        vaklengte = self.length
        wandlengte = self.wandlengte
        totale_directe_bouwkosten_per_meter = c *  wandlengte ** 2 + d * wandlengte + z
        

        self.cost_components['directe_bouwkosten'] = totale_directe_bouwkosten_per_meter * vaklengte



