from dataclasses import dataclass, asdict, field
import concurrent.futures
import requests
import numpy as np
from typing import List, Union
from enum import Enum


class DataTypeAhn(Enum):
    DTM = "dtm"
    DSM = "dsm"


@dataclass
class API_ahn:
    """Spatial utils class for retrieving AHN elevation data."""

    ahn_type: str = "ahn5"
    data_type: DataTypeAhn = DataTypeAhn.DTM
    url_ahn: str = field(init=False)
    request: str = "GetFeatureInfo"
    service: str = "WMS"
    crs: str = "EPSG:28992"
    response_crs: str = "EPSG:28992"
    width: str = "4000"
    height: str = "4000"
    info_format: str = "application/json"
    version: str = "1.3.0"
    layers: str = field(init=False)
    query_layers: str = field(init=False)
    bbox: str = ""
    i: str = "2000"
    j: str = "2000"
    max_workers: int = 50  # safer default
    AHN_data: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    surface_line: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        self.url_ahn = "https://service.pdok.nl/rws/ahn/wms/v1_0"
        if self.ahn_type == "ahn2":
            self.layers = f"{self.ahn_type}_5m"
            self.query_layers = f"{self.ahn_type}_5m"
        else:
            self.layers = "dtm_05m"
            self.query_layers = "dtm_05m"

    @property
    def dictionary_parameters(self):
        """Return parameters for WMS request, excluding internal fields."""
        params = asdict(self)
        for key in ["url_ahn", "ahn_type", "AHN_data", "surface_line", "data_type"]:
            params.pop(key)
        params["data_type"] = self.data_type.value
        return params

    @staticmethod
    def get_ahn_value_from_response(response):
        """Parse AHN WMS response to extract elevation value."""
        if response.status_code != 200:
            raise ConnectionError("Failed to connect to PDOK AHN service.")

        data = response.json()
        features = data.get("features", [])
        if not features:
            return None
        properties = features[0].get("properties", {})
        return float(properties.get("value_list", np.nan))

    def get_ahn_list(self, points_list: Union[List, np.ndarray]):
        """Return a list with [x, y, elevation] for each point."""
        points_array = np.array(points_list)
        if points_array.ndim != 2 or points_array.shape[1] != 2:
            raise ValueError(f"Points must be of shape (n, 2), got {points_array.shape}")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.collect_ahn_data, tuple(pt)) for pt in points_array]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

        self.surface_line = results
        self.AHN_data = np.array([pt for pt in self.surface_line if -7 <= pt[2] <= 323])
        self.AHN_data = self.AHN_data[self.AHN_data[:, 0].argsort()]  # sort by x-coordinate

    def collect_ahn_data(self, point: tuple):
        """Retrieve AHN elevation for a single point."""
        self.bbox = f"{point[0]-1000},{point[1]-1000},{point[0]+1000},{point[1]+1000}"
        response = requests.get(self.url_ahn, params=self.dictionary_parameters)
        value = self.get_ahn_value_from_response(response)
        # print(f"Point AHN value at ({point[0]}, {point[1]}) is {value}")
        if value is not None:
            return [point[0], point[1], value]
        return None
