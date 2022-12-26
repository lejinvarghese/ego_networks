# -*- coding: utf-8 -*-
import pytest
from pandas import DataFrame

try:
    from src.recommender import EgoNetworkRecommender
except ModuleNotFoundError:
    from ego_networks.src.recommender import EgoNetworkRecommender

n_samples = 100
measure = "pagerank"
DEFAULT_PROFILE_URL = "https://cpraonline.files.wordpress.com/2014/07/new-twitter-logo-vector-200x200.png"


@pytest.fixture
def sample_measures():

    return DataFrame(
        {
            "measure_name": [measure for _ in range(n_samples)],
            "node": [str(i) for i in range(n_samples)],
            "measure_value": [i / 10 for i in range(n_samples)],
        }
    )


@pytest.fixture
def sample_targets():
    return {str(i): "name" + "_" + str(i) for i in range(n_samples // 2)}




def test_recommender_train(sample_measures):
    recommender = EgoNetworkRecommender(network_measures=sample_measures)

    results = recommender.train(recommendation_strategy="diverse")

    actual = results.iloc[-1, -1]
    expected = 1 / n_samples
    assert actual == expected

    expected = 1.0
    actual = results.iloc[0, -1]
    assert actual == expected


def test_recommender_test(sample_measures, sample_targets):
    recommender = EgoNetworkRecommender(network_measures=sample_measures)
    recommender.train(recommendation_strategy="diverse")
    precision, recall = recommender.test(sample_targets)
    assert precision == 0.5
    assert recall == 1.0


def test_recommender_recommendations(
    sample_measures, sample_targets,
):
    recommender = EgoNetworkRecommender(network_measures=sample_measures)
    recommender.train(recommendation_strategy="diverse")
    (
        recommended_profile_ids
    ) = recommender.get_recommendations(
        sample_targets, k=3
    )
    assert recommended_profile_ids[0] == "99"
    assert recommended_profile_ids[1] == "98"
    assert recommended_profile_ids[2] == "97"
