# -*- coding: utf-8 -*-
import os

import pytest
from networkx import DiGraph
from pandas import DataFrame

try:
    from src.twitter_network import TwitterEgoNeighborhood
except ModuleNotFoundError:
    from ego_networks.src.twitter_network import TwitterEgoNeighborhood

from dotenv import load_dotenv

load_dotenv()

TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")
MAX_RADIUS = 1
sample_test_twitter_user_names = ["elonmusk", "bulicny"]


@pytest.fixture
def twitter_network():
    return TwitterEgoNeighborhood(
        focal_node=sample_test_twitter_user_names[0],
        max_radius=MAX_RADIUS,
        api_bearer_token=TWITTER_API_BEARER_TOKEN,
        storage_bucket=CLOUD_STORAGE_BUCKET,
    )


@pytest.fixture
def sample_node_features():
    return DataFrame(
        {
            "id": [999, 777],
            "name": ["a", "b"],
            "username": ["us", "ut"],
        }
    )


@pytest.fixture
def sample_edges():
    return DataFrame(
        {"user": [999, 777, 999], "following": [777, 999, 111]},
    )


def test_create_network(twitter_network, sample_edges, sample_node_features):
    actual = twitter_network.create_network(
        sample_edges, sample_node_features, sample=True
    )
    assert type(actual) == DiGraph
    assert actual.number_of_nodes() > 0
    assert actual.number_of_edges() > 0
