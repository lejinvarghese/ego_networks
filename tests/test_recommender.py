# -*- coding: utf-8 -*-
import pytest
from networkx import erdos_renyi_graph
from pandas import DataFrame

try:
    from src.recommender import EgoNetworkRecommender
except ModuleNotFoundError:
    from ego_networks.src.recommender import EgoNetworkRecommender

n_samples = 100


@pytest.fixture
def sample_measures():
    return DataFrame(
        {
            "measure_name": ["pagerank" for _ in range(n_samples)],
            "node": [str(i) for i in range(n_samples)],
            "measure_value": [i / 10 for i in range(n_samples)],
        }
    )


def test_recommender_train(sample_measures):
    recommender = EgoNetworkRecommender(network_measures=sample_measures)

    results = recommender.train()

    actual = results.measure_value.iloc[-1, -1]
    expected = 1 / n_samples
    assert actual == expected

    expected = 1.0
    actual = results.measure_value.iloc[0, -1]
    assert actual == expected
