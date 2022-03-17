import os
import sys
import tqdm
from pathlib import Path
from tqdm.notebook import trange, tqdm
from warnings import filterwarnings

filterwarnings("ignore")

PATH = os.getcwd()
PROJECT = str(Path(PATH).parents[0])


def sample_graph():
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("Joe", "Mary"),
            ("Tom", "Mary"),
            ("Poutine", "Cheese"),
            ("Poutine", "Gravy"),
            ("Potato", "Vegetable"),
            ("Poutine", "Potato"),
            ("Tom", "Potato"),
            ("Gravy", "Meat"),
            ("Tom", "Vegetarian"),
            ("Tom", "Cheese"),
            ("Mary", "Cheese"),
            ("Joe", "Cheese"),
            ("Amazon", "Workplace"),
            ("Tom", "Amazon"),
            ("Joe", "Amazon"),
            ("Tom", "Joe"),
            ("Mary", "Amazon"),
            ("Mary", "Tom"),
            ("Mary", "Vegetarian"),
        ],
        length=8,
    )
    edge_labels = {
        ("Joe", "Mary"): "loves",
        ("Tom", "Mary"): "sibling",
        ("Poutine", "Cheese"): "contains",
        ("Poutine", "Gravy"): "contains",
        ("Potato", "Vegetable"): "is",
        ("Poutine", "Potato"): "contains",
        ("Tom", "Potato"): "likes",
        ("Tom", "Cheese"): "likes",
        ("Joe", "Cheese"): "likes",
        ("Mary", "Cheese"): "likes",
        ("Gravy", "Meat"): "contains",
        ("Tom", "Vegetarian"): "is",
        ("Amazon", "Workplace"): "is",
        ("Tom", "Amazon"): "works",
        ("Joe", "Amazon"): "works",
        ("Mary", "Amazon"): "works",
        ("Tom", "Joe"): "friends",
        ("Tom", "Joe"): "colleagues",
        ("Mary", "Vegetarian"): "is",
    }
    node_labels = {node: node for node in graph.nodes()}
    return graph, edge_labels, node_labels


def draw_graph(
    graph=None,
    edge_labels=None,
    pos=None,
    edge_color="black",
    width=1,
    linewidths=1,
    node_size=3500,
    node_color="black",
    alpha=0.9,
    font_color="w",
    font_size=16,
    fig_size=(20, 20),
    length=17,
    edge_label_font_color="red",
    title=None,
):
    """
    takes a graph and edge labels and draws a diagram. You need to pass edge labels
    if you are passing a graph parameters.
    For smaller subgraphs pass nx.circular_layout for pos.
    """

    if pos == None:
        pos = nx.spring_layout(graph)
    else:
        pos = pos
    fig = plt.figure(figsize=fig_size)
    if title != None:
        fig.suptitle(title)
    nx.draw(
        graph,
        pos=pos,
        edge_color=edge_color,
        width=width,
        linewidths=linewidths,
        node_size=node_size,
        node_color=node_color,
        alpha=alpha,
        font_color=font_color,
        font_size=font_size,
        labels={node: node for node in graph.nodes()},
    )

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_color=edge_label_font_color,
        font_size=font_size,
    )
    plt.axis("off")
    plt.show()


def get_ego_graph(graph, edge_labels, node="Tom", radius=1):
    ego_graph = nx.ego_graph(graph, n=node, radius=radius)
    ego_edge_labels = dict()
    ego_node_labels = {node: node for node in ego_graph.nodes()}
    keys = edge_labels.keys()
    for key in keys:
        if key in ego_graph.edges():
            ego_edge_labels[key] = edge_labels[key]
    return ego_graph, ego_edge_labels, ego_node_labels


def draw_interaction_graph(
    graph, pos=None, first_color="red", second_color="black", font_color="w"
):
    color_map = [second_color for i in range(graph.number_of_nodes())]
    color_map[: graph.number_of_nodes()] = [
        first_color for i in range(graph.number_of_nodes())
    ]
    fig = plt.figure(figsize=(10, 10))
    pos = nx.circular_layout(graph) if pos == "c" else nx.kamada_kawai_layout(graph)
    nx.draw(
        graph,
        with_labels=True,
        pos=pos,
        node_color=color_map,
        font_color=font_color,
        font_weight="bold",
        edge_color=color_map,
    )
    plt.show()
