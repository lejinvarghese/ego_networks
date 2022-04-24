# -*- coding: utf-8 -*-
import os

import pytest
from networkx import DiGraph
from pandas import DataFrame

try:
    from src.network import HomogenousEgoNetwork
except ModuleNotFoundError:
    from ego_networks.src.network import HomogenousEgoNetwork

from dotenv import load_dotenv

load_dotenv()

TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
INTEGRATED_FOCAL_NODE_ID = os.getenv("INTEGRATED_FOCAL_NODE_ID")
RADIUS = 1
sample_test_twitter_user_names = ["elonmusk", "bulicny"]


@pytest.fixture
def twitter_network():
    return HomogenousEgoNetwork(
        focal_node_id=INTEGRATED_FOCAL_NODE_ID,
        radius=RADIUS,
        storage_bucket=None,
    )


@pytest.fixture
def sample_node_features():
    return DataFrame(
        {
            "id": [999, 777, int(INTEGRATED_FOCAL_NODE_ID)],
            "name": ["a", "b", "c"],
            "username": ["us", "ut", "ty"],
        }
    )


@pytest.fixture
def sample_edges():
    return DataFrame(
        {
            "user": [999, 777, 999, int(INTEGRATED_FOCAL_NODE_ID)],
            "following": [777, 999, 111, 111],
        },
    )


def test_create_network(twitter_network, sample_edges, sample_node_features):
    actual = twitter_network.create(
        edges=sample_edges,
        nodes=sample_node_features,
    )
    assert type(actual) == DiGraph
    assert actual.number_of_nodes() > 0
    assert actual.number_of_edges() > 0
