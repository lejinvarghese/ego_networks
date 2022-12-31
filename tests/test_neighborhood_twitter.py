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
def test_twitter_authentication(twitter_client, user_id, expected):
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


def test_neighborhood_instantiation(twitter_neighborhood):
    assert twitter_neighborhood.focal_node == sample_test_twitter_user_names[0]
    assert twitter_neighborhood.max_radius == MAX_RADIUS


def test_update_ties(twitter_neighborhood):
    new_ties, alters_all = twitter_neighborhood.update_ties()
    assert type(new_ties) == DataFrame
    assert type(alters_all) == set
    assert new_ties.shape[0] >= 0
    assert new_ties.shape[1] > 0
    assert len(alters_all) >= 0


def test_get_node_features(twitter_neighborhood, sample_nodes):
    actual = twitter_neighborhood.update_node_features(nodes=sample_nodes)
    feature_fields = [
        "id",
        "name",
        "profile_image_url",
        "public_metrics",
        "username",
        "verified",
    ]

    assert type(actual) == DataFrame
    assert set(actual.columns.values) == set(feature_fields)
    assert actual.shape[0] > 0
    assert actual.shape[1] == len(feature_fields)


def test_update_tie_features(twitter_neighborhood):
    actual = twitter_neighborhood.update_tie_features()
    assert actual is None


def test_delete_ties(twitter_neighborhood):
    cleansed_ties, cleansed_node_features = twitter_neighborhood.delete_ties()
    assert type(cleansed_ties) == DataFrame
    assert type(cleansed_node_features) == DataFrame
    assert cleansed_ties.shape[1] > 0
    assert cleansed_node_features.shape[1] > 0
