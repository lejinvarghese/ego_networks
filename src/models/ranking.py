"""
Ranking algorithms
"""

import pandas as pd
import numpy as np
from scipy.stats import gmean

try:
    from src.models.strategies import weights
    from utils.custom_logger import CustomLogger

except ModuleNotFoundError:
    from ego_networks.src.models.strategies import weights
    from ego_networks.utils.custom_logger import CustomLogger

logger = CustomLogger(__name__)


class WeightedMeasures:
    """
    A class that calculates recommendations from a weighted combination of network measures on the ego network.
    """

    def __init__(
        self,
        data: pd.DataFrame = pd.DataFrame(),
        recommendation_strategy: str = "diverse",
    ):
        self._data = data
        self._weights = weights.get(recommendation_strategy)

    def rank(self):
        X = self._data.copy()
        measures = X.columns.values
        for i in measures:
            X[i] = X[i].rank(pct=True, method="dense")

        measure_weights = np.ones(X.shape[1])
        for i, d in enumerate(self._weights.items()):
            k, v = d
            idx = np.argwhere(measures == k)
            measure_weights[idx] = v

        X["predicted_rank"] = X.iloc[:, -len(measures) :].apply(
            gmean, weights=measure_weights, axis=1
        )
        return X.sort_values(by="predicted_rank", ascending=False)
