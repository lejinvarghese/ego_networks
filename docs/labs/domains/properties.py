def get_graph_properties(graph, random_state=34):
    from community import best_partition
    from networkx import get_edge_attributes, betweenness_centrality

    communities = best_partition(graph, resolution=0.7, random_state=random_state)

    community_center_colors = {
        "Smart Cities": "mediumseagreen",
        "Intelligence": "gold",
        "Robotics": "cornflowerblue",
        "Machine Learning": "deepskyblue",
        "Evolutionary Theory": "mediumpurple",
        "Network Science": "hotpink",
        "Recommender Systems": "lightcoral",
        "Marketing": "palevioletred",
    }

    community_colors = {
        communities.get(k): v for k, v in community_center_colors.items()
    }
    node_colors = [
        community_colors.get(v, "lightgrey") for k, v in communities.items()
    ]

    node_centrality = betweenness_centrality(
        graph,
        weight="weight",
        normalized=True,
        endpoints=True,
        seed=random_state,
    )
    node_sizes = [5e4 * i for i in list(node_centrality.values())]

    edge_colors = []
    for e in graph.edges():
        u, v = e
        if communities.get(u) == communities.get(v):
            edge_colors.append(
                community_colors.get(communities.get(u), "lightgrey")
            )
        else:
            edge_colors.append("lightgrey")

    edge_sizes = [
        i * 1e1 for i in get_edge_attributes(graph, "weight").values()
    ]

    return node_colors, node_sizes, edge_colors, edge_sizes
