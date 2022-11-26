# -*- coding: utf-8 -*-
import pytest
from networkx import erdos_renyi_graph
from pandas import DataFrame

try:
    from src.measures import EgoNetworkMeasures
except ModuleNotFoundError:
    from ego_networks.src.measures import EgoNetworkMeasures

n_nodes = 100


@pytest.fixture
def sample_graph():
    return erdos_renyi_graph(
        n=n_nodes,
        p=0.1,
        directed=True,
        seed=42,
    )


def test_summary_measures(sample_graph):
    actual = EgoNetworkMeasures(sample_graph).summary_measures

    assert type(actual) == DataFrame
    assert actual.shape[0] > 0
    assert actual[actual.index == "n_nodes"].iloc[0, 0] == n_nodes
    assert actual[actual.index == "pairs"].iloc[0, 0] >= n_nodes
    assert actual.shape[1] > 0


def test_node_measures(sample_graph):
    actual = EgoNetworkMeasures(
        sample_graph, calculate_nodes=True
    ).node_measures

    assert type(actual) == DataFrame
    assert actual.shape[0] >= n_nodes
    assert set(actual.reset_index().measure_name.unique()) == set(
        [
            "degree_centrality",
            "closeness_centrality",
            "betweenness_centrality",
            "eigenvector_centrality",
            "pagerank",
            "hubs",
            "authorities",
        ]
    )
