import geopandas as gpd

from app.unit_costs_and_surcharges import load_kosten_catalogus
from ..AHN_raster_API import AHN4_API
from ..projection_transformation import transform_to_rd
import numpy as np

class StructureModel:
    valid_constructietypes = ['Onverankerde damwand', 'Verankerde damwand', 'Heavescherm']

    '''Model for calculating properties of structures. Structure should contain of one single line segment with properties.'''
    def __init__(self, location: gpd.GeoDataFrame, complexity: str = 'gemiddeld'):
        if len(location) != 1:
            raise ValueError("Location must contain exactly one line segment.")
        self.location = location
        # Transform coordinates to RD
        self.location = self.location.to_crs(epsg=28992)
        
        self.complexity = complexity
        
        #check if all types in location are the same
        types = self.location['type'].unique()
        if len(types) > 1:
            raise ValueError("Meerdere constructietypes in 1 lijnsegment. Dit is niet toegestaan.")
        
        
        self.constructietype = types[0]

        if self.constructietype not in self.valid_constructietypes:
            raise ValueError(f"Invalid constructietype: {self.constructietype}. Must be one of {self.valid_constructietypes}.")
        
        self.diepte = self.location['diepte'].iloc[0]


        # Determine length of the structure by importing the AHN for the line segment
        self.length = self.location.geometry.length.sum()
        
        self.determine_length_from_depth()

        self.get_screen_length()

        self.cost_catalog = load_kosten_catalogus()
        
    def determine_length_from_depth(self, ahn_type = 'AHN4'):
        '''Determine the length of the structure based on depth and AHN data.'''
        # Placeholder for AHN data processing
        # In a real implementation, this would involve querying AHN data
        # and calculating the length of the elements based on depth values.
        self.elevation = AHN4_API().get_elevation_from_line(self.location.geometry[0])

    def get_screen_length(self, type = 'mean'):
        if not hasattr(self, 'elevation'):
            raise ValueError("Elevation data not available. Please run determine_length_from_depth() first.")
        
        # Calculate the length of the structure that is below the specified depth

        #get Z values from elevation LineString
        Z = [self.elevation.coords[i][2] for i in range(len(self.elevation.coords))]
        if type == 'mean':
            top_level = np.mean(Z) 
            self.wandlengte = max(0, top_level - self.diepte)
        else:
            raise ValueError("Niet geimplementeerd voor andere types dan 'mean'.")

    def compute_directe_bouwkosten(self, c, d, z) -> float:
        raise NotImplementedError("This method should be implemented in subclasses.")

        
    

