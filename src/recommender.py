"""
Relevant documentation
"""

import pandas as pd
from scipy.stats import gmean

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

        if network_measures is not None:
            self.network_measures = network_measures
        elif (network_measures is None) & (self._storage_bucket is not None):
            self.network_measures = read_data(
                self._storage_bucket, "node_measures"
            )
        else:
            raise ValueError(
                "Either network_measures or path of the storage_bucket that contain network_measures must be provided"
            )

    def train(self):

        scores = self.network_measures.pivot(
            index="node", columns="measure_name"
        )
        scores["measure_value"] = scores["measure_value"].fillna(value=0)

        measures = scores.columns.values
        for i in measures:
            scores[i] = scores[i].rank(pct=True, method="dense")
        scores["rank_combined"] = scores.iloc[:, -len(measures) :].apply(
            gmean, axis=1
        )
        scores = scores.sort_values(by="rank_combined", ascending=False)
        self.__model = scores
        return scores

    def test(self, targets: dict, k: int = 100):
        predicted = set(self.__model.index.to_list()[:k])
        actuals = set(list(targets.keys()))
        true_positive = predicted.intersection(actuals)
        precision_k = round(len(true_positive) / len(predicted), 2)
        recall_k = round(len(true_positive) / min(k, len(actuals)), 2)
        print(f"Precision @ k: {k}: {precision_k}")
        print(f"Recall @ k: {k}: {recall_k}")
        return precision_k, recall_k

    def get_recommendations(self, targets: dict, labels: dict, k: int = 10):
        """
        Returns a list of recommendations for the focal node.
        """
        predicted = self.__model.index.to_list()
        actuals = list(targets.keys())
        recs = [labels.get(i) for i in predicted if i not in actuals]
        return recs[:k]
