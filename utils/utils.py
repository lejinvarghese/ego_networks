import os
from pathlib import Path
from warnings import filterwarnings

filterwarnings("ignore")

PATH = os.getcwd()
PROJECT = str(Path(PATH).parents[0])


def sample_knowledge_graph():
    import networkx as nx

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


def draw_nx_graph(
    graph=None,
    pos=None,
    node_labels=None,
    edge_labels=None,
    width=1,
    linewidths=1,
    node_size=3500,
    alpha=0.9,
    font_size=16,
    fig_size=(20, 20),
    dpi=240,
    length=17,
    node_color="dimgrey",
    node_label_font_color="whitesmoke",
    edge_color="peachpuff",  # "darkgrey",
    edge_label_font_color="tomato",
    title=None,
):
    """
    Takes a graph and edge labels and draws a diagram. You need to pass edge labels
    if you are passing a graph parameters.
    For smaller subgraphs pass nx.circular_layout for pos.
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    if pos:
        _pos = pos
    else:
        _pos = nx.spring_layout(graph)

    if node_labels:
        _with_labels = True
    else:
        _with_labels = False

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    if title:
        fig.suptitle(title)
    nx.draw(
        graph,
        pos=_pos,
        edge_color=edge_color,
        width=width,
        linewidths=linewidths,
        node_size=node_size,
        node_color=node_color,
        alpha=alpha,
        font_color=node_label_font_color,
        font_size=font_size,
        with_labels=_with_labels,
    )

    nx.draw_networkx_edge_labels(
        graph,
        _pos,
        edge_labels=edge_labels,
        font_color=edge_label_font_color,
        font_size=font_size,
    )
    plt.axis("off")
    plt.show()


def get_ego_graph(graph, edge_labels=None, node="Tom", radius=1):
    import networkx as nx

    ego_graph = nx.ego_graph(graph, n=node, radius=radius)
    ego_edge_labels = dict()
    ego_node_labels = {node: node for node in ego_graph.nodes()}
    if edge_labels:
        keys = edge_labels.keys()
        for key in keys:
            if key in ego_graph.edges():
                ego_edge_labels[key] = edge_labels[key]
    return ego_graph, ego_node_labels, ego_edge_labels


def draw_interaction_graph(
    graph, pos=None, first_color="red", second_color="black", font_color="w"
):
    import networkx as nx
    import matplotlib.pyplot as plt

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


def draw_plotly_graph(
    graph=None,
    edge_labels=None,
    node_labels=None,
    pos=None,
    node_colors=None,
    node_sizes=None,
    write_html=False,
    title=None,
    fig_size_px=(800, 800),
    hide_color_axis=True,
):

    import plotly.graph_objects as go
    import networkx as nx

    edge_x = []
    edge_y = []
    pos = nx.spring_layout(graph, threshold=1e-5)
    c_ev = nx.eigenvector_centrality(graph)
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale="Bluered",
            reversescale=False,
            color=[],
            size=10,
            colorbar=dict(thickness=5, xanchor="left", titleside="right"),
            line_width=0.0,
        ),
    )

    # color nodes
    if node_labels:
        _node_labels = node_labels
    else:
        node_adjacencies = []
        _node_labels = []
        for node, adjacencies in enumerate(graph.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            _node_labels.append("adjacent connections: " + str(len(adjacencies[1])))
    node_trace.text = _node_labels

    if node_colors == "default":
        c_ev = nx.eigenvector_centrality(graph)
        _node_colors = [1e2 * x for x in list(c_ev.values())]
    else:
        _node_colors = node_colors
    node_trace.marker.color = _node_colors

    if node_sizes == "default":
        c_ev = nx.eigenvector_centrality(graph)
        _node_sizes = [1e2 * x for x in list(c_ev.values())]
    else:
        _node_sizes = node_sizes
    node_trace.marker.size = _node_sizes

    # create plot
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title={
                "text": title,
                "font_size": 16,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",  # new
            },
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            width=fig_size_px[0],
            height=fig_size_px[1],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    if hide_color_axis:
        fig.update_coloraxes(showscale=False)

    fig.show()

    if write_html:
        fig.write_html("first_degree_test.html", auto_open=True)
