"""
Reference:
1. https://faculty.ucr.edu/~hanneman/nettext/C9_Ego_networks.html
2. networkx: https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html
3. brokerage: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5325703/
"""


import networkx as nx
import pandas as pd
import time

try:
    from src.core import NetworkMeasures
except ModuleNotFoundError:
    from ego_networks.src.core import NetworkMeasures


class EgoNetworkMeasures(NetworkMeasures):
    """ """

    def __init__(self, G, nodes=False, edges=False):
        self.G = G
        self.nodes = nodes
        self.edges = edges

    @property
    def summary_measures(self):
        return self.__create_graph_measures()

    @property
    def node_measures(self):
        if self.nodes:
            return self.__create_node_measures()
        else:
            raise ValueError("Node measures not available.")

    @property
    def edge_measures(self):
        if self.edges:
            return self.__create_edge_measures()
        else:
            raise ValueError("Edge measures not available.")

    def __create_graph_measures(self):
        measures = {}
        measures["n_nodes"] = measures["size"] = len(self.G.nodes())
        measures["n_edges"] = measures["ties"] = len(self.G.edges())
        measures["pairs"] = measures.get("size") * (measures.get("size") - 1)
        measures["density"] = nx.density(self.G)

        measures["transitivity"] = nx.transitivity(self.G)
        measures["average_clustering"] = nx.average_clustering(self.G)
        measures[
            "n_strongly_connected_components"
        ] = nx.number_strongly_connected_components(self.G)
        measures["n_attracting_components"] = nx.number_attracting_components(
            self.G
        )
        measures["global_reaching_centrality"] = nx.global_reaching_centrality(
            self.G
        )

        return (
            pd.DataFrame.from_dict(
                measures, orient="index", columns=["measure_value"]
            )
            .rename_axis(index="measure_name")
            .round(4)
            .sort_index()
        )

    def __create_node_measures(self):
        measures = {}
        measures["degree_centrality"] = nx.in_degree_centrality(self.G)
        measures["betweenness_centrality"] = nx.betweenness_centrality(
            self.G, k=min(len(self.G.nodes()), 500)
        )

        measures["eigenvector_centrality"] = nx.eigenvector_centrality_numpy(
            self.G
        )
        measures["pagerank"] = nx.pagerank_scipy(self.G)
        measures["hubs"], measures["authorities"] = nx.hits(self.G)
        measures["closeness_centrality"] = nx.closeness_centrality(self.G)
        return (
            pd.DataFrame.from_dict(measures, orient="index")
            .rename_axis(index="measure_name")
            .melt(
                ignore_index=False,
                var_name="node",
                value_name="measure_value",
            )
            .round(4)
            .reset_index()
            .sort_values(
                by=["measure_name", "measure_value"],
                ascending=False,
            )
        )

    def __create_edge_measures(self):
        measures = {}
        # start = time.time()
        measures[
            "edge_betweenness_centrality"
        ] = nx.edge_betweenness_centrality(
            self.G, k=min(len(self.G.nodes()), 100), seed=42
        )
        # end = time.time()
        # print(f"Function took {end - start} seconds.")
        for k, v in measures.items():
            measures[k] = (
                pd.Series(v)
                .rename_axis(["source_node", "target_node"])
                .reset_index(name="measure_value")
            )

        return (
            pd.concat(measures, axis=0)
            .reset_index(level=0)
            .rename({"level_0": "measure_name"}, axis=1)
            .sort_values(
                by=["measure_name", "measure_value"],
                ascending=False,
            )
        )
