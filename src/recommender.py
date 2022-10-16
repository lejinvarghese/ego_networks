"""
Relevant documentation
"""

import pandas as pd
import numpy as np
from scipy.stats import gmean

try:
    from src.core import NetworkRecommender
    from utils.io import DataReader
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.src.core import NetworkRecommender
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

    def train(self, weights: dict = None):

        scores = self.network_measures.pivot(
            index="node", columns="measure_name"
        )
        scores["measure_value"] = scores["measure_value"].fillna(value=0)

        measures = scores.columns.values
        for i in measures:
            scores[i] = scores[i].rank(pct=True, method="dense")

        measure_weights = np.ones(scores.shape[1])
        weights = {"brokerage": 5, "pagerank": 2}
        for k, v in weights.items():
            k_t = ("measure_value", k)
            idx = np.argwhere(measures == k_t).flatten()
            measure_weights[idx] = v

        measure_weights[-1] = 5
        scores["rank_combined"] = scores.iloc[:, -len(measures) :].apply(
            gmean, weights=measure_weights, axis=1
        )
        scores = scores.sort_values(by="rank_combined", ascending=False)
        self.__model = scores
        return scores

    def test(self, targets: dict, k: int = 100):
        predicted = set(self.__model.index.to_list()[:k])
        actuals = set(list(targets.keys()))
        true_positive = predicted.intersection(actuals)
        precision_k = round(len(true_positive) / len(predicted), 2)
        print(len(actuals), k)
        recall_k = round(len(true_positive) / min(k, len(actuals)), 2)
        logger.info(f"Precision @ k: {k}: {precision_k}")
        logger.info(f"Recall @ k: {k}: {recall_k}")
        return precision_k, recall_k

    def get_recommendations(self, targets: dict, labels: dict, k: int = 10):
        """
        Returns a list of recommendations for the focal node.
        """
        predicted = self.__model.index.to_list()
        actuals = list(targets.keys())
        recs = [labels.get(i) for i in predicted if i not in actuals]
        return recs[:k]
