# -*- coding: utf-8 -*-
import os

import pytest
from pandas import DataFrame

try:
    from src.neighborhoods.goodreads import GoodreadsEgoNeighborhood
    from utils.api.goodreads import get_shelf_data
except ModuleNotFoundError:
    from ego_networks.src.neighborhoods.goodreads import (
        GoodreadsEgoNeighborhood,
    )
    from ego_networks.utils.api.goodreads import get_shelf_data

from dotenv import load_dotenv

load_dotenv()

GOODREADS_FOCAL_NODE_ID = os.getenv("GOODREADS_FOCAL_NODE_ID")
MAX_RADIUS = 1
sample_shelves = ["currently-reading", "to-read"]


@pytest.fixture
def goodreads_neighborhood():
    return GoodreadsEgoNeighborhood(
        focal_node=GOODREADS_FOCAL_NODE_ID,
        max_radius=MAX_RADIUS,
    )


@pytest.fixture
def sample_shelf_data():
    return DataFrame(
        {
            "title": ["xyz", "xxz"],
            "author": ["a", "b"],
            "date": ["2022-01-01", "2020-01-02"],
            "shelf": ["currently-reading", "to-read"],
        }
    )


def test_get_shelf_data():
    actual = get_shelf_data(
        user_id=GOODREADS_FOCAL_NODE_ID, shelf=sample_shelves[-1]
    )
    expected_feature_fields = [
        "title",
        "author",
        "date",
        "shelf",
    ]

    assert type(actual) == DataFrame
    assert set(actual.columns.values) == set(expected_feature_fields)
    assert actual.shape[0] > 0
    assert actual.shape[1] == len(expected_feature_fields)


def test_update_neighborhood(goodreads_neighborhood):
    (
        actual_node_features,
        actual_ties,
    ) = goodreads_neighborhood.update_neighborhood()

    expected_node_feature_fields = [
        "title",
        "author",
        "date",
        "shelf",
    ]
    expected_tie_fields = ["source", "target", "weight"]

    assert type(actual_node_features) == DataFrame
    assert type(actual_ties) == DataFrame
    assert set(actual_node_features.columns.values) == set(
        expected_node_feature_fields
    )
    assert actual_node_features.shape[0] > 0
    assert actual_ties.shape[0] > 0
    assert actual_ties.shape[1] == len(expected_tie_fields)
