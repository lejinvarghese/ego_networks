# -*- coding: utf-8 -*-
import pytest
from pandas import DataFrame

try:
    from src.models.ranking import WeightedMeasures
    from src.models.strategies import weights
except ModuleNotFoundError:
    from ego_networks.src.models.ranking import WeightedMeasures
    from ego_networks.src.models.strategies import weights

n_samples = 100
measure = "pagerank"


@pytest.fixture
def sample_data():

    return DataFrame(
        {
            "measure_name": [measure for _ in range(n_samples)],
            "node": [str(i) for i in range(n_samples)],
            "measure_value": [i / 10 for i in range(n_samples)],
        }
    )


@pytest.fixture
def sample_weighted_measures(sample_data):
    return WeightedMeasures(sample_data)


@pytest.fixture
def sample_strategy_weights():
    return weights


def test_strategies(sample_strategy_weights):
    actual = set(list(sample_strategy_weights.keys()))
    expected = set(["diverse", "connectors", "influencers"])
    assert actual == expected


def test_strategy_weights(sample_strategy_weights):
    actual = list(sample_strategy_weights.get("connectors").values())
    for a in actual:
        assert type(a) == int


def test_weighted_measures_model(sample_weighted_measures):
    actual = sample_weighted_measures.rank()
    assert type(actual) == DataFrame
    assert actual.shape[0] == n_samples
    assert "predicted_rank" in actual.columns.values
    assert actual.iloc[0]["predicted_rank"] == 1
