# -*- coding: utf-8 -*-
import os

import pytest
from networkx import DiGraph
from pandas import DataFrame

try:
    from src.twitter_network import TwitterEgoNetwork
except ModuleNotFoundError:
    from ego_networks.src.twitter_network import TwitterEgoNetwork

from dotenv import load_dotenv

load_dotenv()

TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
TEST_TWITTER_IDS = [44196397, 2622261]
TEST_TWITTER_USERNAMES = ["elonmusk", "bulicny"]
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")
NETWORK_RADIUS = 1


@pytest.fixture
def twitter_network():
    return TwitterEgoNetwork(
        focal_node=TEST_TWITTER_USERNAMES[0],
        max_radius=NETWORK_RADIUS,
        api_bearer_token=TWITTER_API_BEARER_TOKEN,
        storage_bucket=CLOUD_STORAGE_BUCKET,
    )


def test_instantiation(twitter_network):
    assert twitter_network.focal_node == TEST_TWITTER_USERNAMES[0]
    assert twitter_network.max_radius == NETWORK_RADIUS


@pytest.mark.parametrize(
    "user_id, expected",
    [
        (TEST_TWITTER_IDS[0], TEST_TWITTER_USERNAMES[0]),
        (TEST_TWITTER_IDS[1], TEST_TWITTER_USERNAMES[1]),
    ],
)
def test_authentication(twitter_network, user_id, expected):
    actual = (
        twitter_network.client.get_users(
            ids=[user_id], user_fields=["username"]
        )
        .data[0]
        .username
    )
    assert actual == expected


def test_retrieve_node_features_id(twitter_network):
    actual = twitter_network.retrieve_node_features(
        user_fields=["id"], user_names=[TEST_TWITTER_USERNAMES[0]]
    )[0].id
    assert actual == TEST_TWITTER_IDS[0]


def test_retrieve_node_features_absent(twitter_network):
    with pytest.raises(ValueError):
        twitter_network.retrieve_node_features(user_fields=["id"])


def test_retrieve_alter_features(twitter_network):
    actual = twitter_network.update_alter_features(
        alters=[999, 9999, 99999, 77777]
    )
    alter_return_fields = [
        "id",
        "name",
        "profile_image_url",
        "public_metrics",
        "username",
        "verified",
        "withheld",
    ]

    assert type(actual) == DataFrame
    assert set(actual.columns.values) == set(alter_return_fields)
    assert actual.shape[0] > 0
    assert actual.shape[1] == len(alter_return_fields)


def test_create_network(twitter_network):
    actual = twitter_network.create_network()
    assert type(actual) == DiGraph
    assert actual.number_of_nodes() > 0
    assert actual.number_of_edges() > 0
