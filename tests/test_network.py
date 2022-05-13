# -*- coding: utf-8 -*-
import pytest
from networkx import DiGraph
from pandas import DataFrame

try:
    from src.network import HomogenousEgoNetwork
except ModuleNotFoundError:
    from ego_networks.src.network import HomogenousEgoNetwork

RADIUS = 1
sample_test_twitter_user_names = ["elonmusk", "bulicny"]


@pytest.fixture
def sample_node_features():
    return DataFrame(
        {
            "id": [999, 777, 888],
            "name": ["a", "b", "c"],
            "username": ["us", "ut", "ty"],
        }
    )


@pytest.fixture
def sample_edges():
    return DataFrame(
        {
            "user": [999, 777, 999, 888],
            "following": [777, 999, 111, 111],
        },
    )


@pytest.fixture
def twitter_network(sample_node_features, sample_edges):
    return HomogenousEgoNetwork(
        focal_node_id=999,
        radius=RADIUS,
        nodes=sample_node_features,
        edges=sample_edges,
        storage_bucket=None,
    )


def test_create_network(twitter_network):
    actual = twitter_network.G
    assert type(actual) == DiGraph
    assert actual.number_of_nodes() > 0
    assert actual.number_of_edges() > 0
