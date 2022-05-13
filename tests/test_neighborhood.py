# -*- coding: utf-8 -*-
import os

import pytest
from pandas import DataFrame

try:
    from src.neighborhoods.twitter import TwitterEgoNeighborhood
    from utils.api.twitter import authenticate, get_users, get_users_following
except ModuleNotFoundError:
    from ego_networks.src.neighborhoods.twitter import TwitterEgoNeighborhood
    from ego_networks.utils.api.twitter import (
        authenticate,
        get_users,
        get_users_following,
    )

from dotenv import load_dotenv

load_dotenv()

TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")
MAX_RADIUS = 1
sample_test_twitter_user_ids = [44196397, 2622261]
sample_test_twitter_user_names = ["elonmusk", "bulicny"]


@pytest.fixture
def twitter_client():
    return authenticate(TWITTER_API_BEARER_TOKEN)


@pytest.fixture
def twitter_neighborhood():
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
def sample_nodes():
    return set([999, 9999, 99999, 77777])


@pytest.fixture
def sample_edges():
    return DataFrame(
        {"user": [999, 777, 999], "following": [777, 999, 111]},
    )


@pytest.mark.parametrize(
    "user_id, expected",
    [
        (sample_test_twitter_user_ids[0], sample_test_twitter_user_names[0]),
        (sample_test_twitter_user_ids[1], sample_test_twitter_user_names[1]),
    ],
)
def test_authentication(twitter_client, user_id, expected):
    actual = get_users(
        client=twitter_client, user_ids=[user_id], user_fields=["username"]
    )[0].username
    assert actual == expected


def test_get_users_id(twitter_client):
    actual = get_users(
        client=twitter_client,
        user_fields=["id"],
        user_names=[sample_test_twitter_user_names[0]],
    )[0].id
    assert actual == sample_test_twitter_user_ids[0]


def test_get_users_following(twitter_client):
    actual = get_users_following(
        client=twitter_client, user_id=sample_test_twitter_user_ids[0]
    ).get("following")
    assert len(actual) > 0


def test_get_users_absent(twitter_client):
    with pytest.raises(ValueError):
        get_users(client=twitter_client, user_fields=["id"])


def test_instantiation(twitter_neighborhood):
    assert twitter_neighborhood.focal_node == sample_test_twitter_user_names[0]
    assert twitter_neighborhood.max_radius == MAX_RADIUS


def test_get_node_features(twitter_neighborhood, sample_nodes):
    actual = twitter_neighborhood.update_node_features(nodes=sample_nodes)
    feature_fields = [
        "id",
        "name",
        "profile_image_url",
        "public_metrics",
        "username",
        "verified",
        "withheld",
    ]

    assert type(actual) == DataFrame
    assert set(actual.columns.values) == set(feature_fields)
    assert actual.shape[0] > 0
    assert actual.shape[1] == len(feature_fields)
