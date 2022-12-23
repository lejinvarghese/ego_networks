# communities = {
#     "Aesthetics": 1,
#     "Marketing": 1,
#     "Diversity": 1,
#     "Artificial Life": 4,
#     "Robotics": 4,
#     "Machine Learning": 3,
#     "Astronomy": 4,
#     "Astrophysics": 4,
#     "Space Exploration": 4,
#     "Philosophy": 5,
#     "Psychology": 5,
#     "Computational Social Science": 4,
#     "Causality": 5,
#     "Statistics": 5,
#     "Chaos Theory": 2,
#     "Game Theory": 2,
#     "Complexity Theory": 2,
#     "Cognitive Neuroscience": 5,
#     "Complex Socio-Technical Systems": 2,
#     "Recommender Systems": 2,
#     "Evolutionary Theory": 2,
#     "Optimization Algorithms": 4,
#     "Network Science": 4,
#     "Deep Learning": 3,
#     "Reinforcement Learning": 3,
#     "Electronics": 4,
#     "Internet Of Things": 4,
#     "Intelligence": 5,
#     "Equity": 1,
#     "Ethics": 5,
#     "Politics": 5,
#     "Genetics": 2,
#     "Federated Machine Learning": 3,
#     "Innovation": 1,
#     "Non Linear Dynamics": 2,
#     "Smart Cities": 0,
#     "Urban Mobility": 0,
# }
# community_center_colors = {
#     "Smart Cities": "green",
#     "Intelligence": "gold",
#     "Robotics": "royalblue",
#     "Machine Learning": "salmon",
#     "Complexity Theory": "orchid",
#     "Marketing": "purple",
# }

# community_colors = {
#     communities.get(k): v for k, v in community_center_colors.items()
# }
# node_colors = [community_colors.get(v) for k, v in communities.items()]
# edges = [
#     ("Smart Cities", "Urban Mobility", {"weight": 5}),
#     ("Robotics", "Urban Mobility", {"weight": 5}),
#     ("Marketing", "Equity", {"weight": 5}),
# ]

# edge_colors = []
# for e in edges:
#     u, v, d = e
#     if communities.get(u) == communities.get(v):
#         edge_colors.append(community_colors.get(communities.get(u)))
#     else:
#         edge_colors.append("red")


def get_graph_properties(graph, random_state=34):
    from community import best_partition
    from networkx import betweenness_centrality, get_edge_attributes

    communities = best_partition(graph, resolution=0.8, random_state=42)

    community_center_colors = {
        "Smart Cities": "green",
        "Intelligence": "gold",
        "Robotics": "royalblue",
        "Machine Learning": "salmon",
        "Complexity Theory": "orchid",
        "Marketing": "purple",
    }

    community_colors = {
        communities.get(k): v for k, v in community_center_colors.items()
    }
    node_colors = [community_colors.get(v) for k, v in communities.items()]

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
            edge_colors.append(community_colors.get(communities.get(u)))
        else:
            edge_colors.append("dimgrey")

    edge_sizes = [i*1e1 for i in get_edge_attributes(graph, "weight").values()]

    return node_colors, node_sizes, edge_colors, edge_sizes
