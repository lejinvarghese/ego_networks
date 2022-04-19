import pytest
from src.network import TwitterEgoNetwork

TEST_TWITTER_IDS = [44196397, 2622261]
TEST_TWITTER_USERNAME = ["elonmusk", "bulicny"]
NETWORK_RADIUS = 1


@pytest.fixture
def twitter_network():
    return TwitterEgoNetwork(focal_node=TEST_TWITTER_IDS[0], radius=NETWORK_RADIUS)


def test_instantiation(twitter_network):
    assert twitter_network.focal_node == TEST_TWITTER_IDS[0]
    assert twitter_network.radius == NETWORK_RADIUS


@pytest.mark.parametrize(
    "user_id, expected",
    [
        (TEST_TWITTER_IDS[0], TEST_TWITTER_USERNAME[0]),
        (TEST_TWITTER_IDS[1], TEST_TWITTER_USERNAME[1]),
    ],
)
def test_twitter_authentication(twitter_network, user_id, expected):
    actual = user_id
    assert actual == expected

def test_gcs_authentication(twitter_network):
    pass