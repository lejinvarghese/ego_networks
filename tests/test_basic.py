import pytest
import os

from src.network import TwitterEgoNetwork
from dotenv import load_dotenv

load_dotenv()

TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
TEST_TWITTER_IDS = [44196397, 2622261]
TEST_TWITTER_USERNAMES = ["elonmusk", "bulicny"]
NETWORK_RADIUS = 1


@pytest.fixture
def twitter_network():
    return TwitterEgoNetwork(focal_node=TEST_TWITTER_USERNAMES[0], max_radius=NETWORK_RADIUS)


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
def test_twitter_authentication(twitter_network, user_id, expected):
    tn = twitter_network.authenticate(TWITTER_API_BEARER_TOKEN)
    actual = tn.client.get_users(ids=[user_id], user_fields=["username"]).data[0].username
    assert actual == expected


def test_retrieve_node_features_id(twitter_network):
    tn = twitter_network.authenticate(TWITTER_API_BEARER_TOKEN)
    actual = tn._retrieve_node_features(user_fields=["id"], user_names=[TEST_TWITTER_USERNAMES[0]])[
        0
    ].id
    assert actual == TEST_TWITTER_IDS[0]


def test_retrieve_node_features_absent(twitter_network):
    tn = twitter_network.authenticate(TWITTER_API_BEARER_TOKEN)
    with pytest.raises(ValueError):
        tn._retrieve_node_features(user_fields=["id"])
