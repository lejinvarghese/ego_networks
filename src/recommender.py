"""
Recommender class
"""

import pandas as pd
import numpy as np

try:
    from src.core import NetworkRecommender
    from src.models.ranking import WeightedMeasures
    from utils.io import DataReader
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.src.core import NetworkRecommender
    from ego_networks.src.models.ranking import WeightedMeasures
    from ego_networks.utils.io import DataReader
    from ego_networks.utils.custom_logger import CustomLogger

logger = CustomLogger(__name__)


class EgoNetworkRecommender(NetworkRecommender):
    """
    An object that calculates recommendations from the ego network.
    """

    def __init__(
        self,
        network_measures: pd.DataFrame = pd.DataFrame(),
        max_radius: int = 2,
        use_cache: bool = False,
    ):
        self._max_radius = max_radius
        self._use_cache = use_cache

        if self._use_cache:
            self.network_measures = DataReader(data_type="node_measures").run()
        elif not (self._use_cache):
            self.network_measures = network_measures

        if self.network_measures.empty:
            raise ValueError(
                "Either calculated or cached network_measures should be provided."
            )

    def train(self, recommendation_strategy):

        measures = self.network_measures.pivot(
            index="node", columns="measure_name"
        )
        measures["measure_value"] = measures["measure_value"].fillna(value=0)
        measures.columns = measures.columns.get_level_values(1)
        model = WeightedMeasures(
            data=measures, recommendation_strategy=recommendation_strategy
        )
        self.__ranks = model.rank()
        return np.round(self.__ranks, 2)

    def test(self, targets: dict, k: int = 100):
        predicted = set(self.__ranks.index.to_list()[:k])
        actuals = set(list(targets.keys()))
        true_positive = predicted.intersection(actuals)
        precision_k = round(len(true_positive) / len(predicted), 2)
        recall_k = round(len(true_positive) / min(k, len(actuals)), 2)
        logger.info(f"Precision @ k: {k}: {precision_k}")
        logger.info(f"Recall @ k: {k}: {recall_k}")
        return precision_k, recall_k

    def get_recommendations(
        self, targets: dict, labels: dict, profile_images: dict, k: int = 10
    ):
        """
        Returns a list of recommendations for the focal node.
        """
        predicted = self.__ranks.index.to_list()
        actuals = list(targets.keys())
        recs = [labels.get(i) for i in predicted if i not in actuals][:k]
        profiles = [
            profile_images.get(i) for i in predicted if i not in actuals
        ][:k]
        return recs, profiles
