# -*- coding: utf-8 -*-
from unittest.mock import patch
import pytest
from networkx import erdos_renyi_graph, random_k_out_graph, margulis_gabber_galil_graph

try:
    from utils.graph import draw_nx_graph
except ModuleNotFoundError:
    from ego_networks.utils.graph import draw_nx_graph


def get_sample_graph(graph_type):
    if graph_type == "directed":
        return erdos_renyi_graph(n=20, p=0.1, directed=True)
    elif graph_type == "undirected":
        return erdos_renyi_graph(n=20, p=0.1, directed=False)
    elif graph_type == "cyclic":
        return random_k_out_graph(n=20, k=5, alpha=0.2)
    elif graph_type == "multigraph":
        return margulis_gabber_galil_graph(n=20)

@patch('matplotlib.pyplot.show')
def test_plot_directed(mock_plot):
    graph = get_sample_graph(graph_type="directed")
    draw_nx_graph(graph)

@patch('matplotlib.pyplot.show')
def test_plot_undirected(mock_plot):
    graph = get_sample_graph(graph_type="undirected")
    draw_nx_graph(graph)

@patch('matplotlib.pyplot.show')
def test_plot_cyclic(mock_plot):
    graph = get_sample_graph(graph_type="cyclic")
    draw_nx_graph(graph)

@patch('matplotlib.pyplot.show')
def test_plot_multigraph(mock_plot):
    graph = get_sample_graph(graph_type="multigraph")
    draw_nx_graph(graph)

@patch('matplotlib.pyplot.show')
def test_plot_cyclic_invalid_style(mock_plot):
    graph = get_sample_graph(graph_type="cyclic")
    with pytest.raises(ValueError):
        draw_nx_graph(graph, arrowstyle="wedge")

@patch('matplotlib.pyplot.show')
def test_plot_cyclic_invalid_values(mock_plot):
    graph = get_sample_graph(graph_type="undirected")
    with pytest.raises(ValueError):
        draw_nx_graph(graph, font_size="x")
    with pytest.raises(ValueError):
        draw_nx_graph(graph, node_size="xx")
    with pytest.raises(AttributeError):
        draw_nx_graph(graph, node_labels="xxx")