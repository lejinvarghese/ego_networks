"""
Relevant documentation
"""

import pandas as pd

try:
    from src.core import NetworkRecommender
    from utils.default import read_data
except ModuleNotFoundError:
    from ego_networks.src.core import NetworkRecommender
    from ego_networks.utils.default import read_data


class EgoNetworkRecommender(NetworkRecommender):
    """
    An object that calculates recommendations from the ego network.
    """

    def __init__(
        self,
        storage_bucket: str = None,
        max_radius: int = 2,
        network_measures: pd.DataFrame = None,
    ):
        self._storage_bucket = storage_bucket
        self._max_radius = max_radius

        if self._storage_bucket is not None:
            self.network_measures = read_data(
                self._storage_bucket, "node_measures"
            )
        else:
            self.network_measures = network_measures

    def train(self):
        return self.network_measures.sort_values(
            by="measure_value", ascending=False
        )

    def test(self):
        pass

    def get_recommendations(k: int = 10):
        """
        Returns a list of recommendations for the focal node.
        """
        pass
