# -*- coding: utf-8 -*-
"""
A multi layered complex network spanning social media neighborhoods.
"""


import os
from warnings import filterwarnings

import networkx as nx
import pandas as pd
from dotenv import load_dotenv

try:
    from src.core import EgoNetwork, NetworkMeasures
    from src.neighborhoods.twitter import TwitterEgoNeighborhood
    from utils.generic import read_data
except ModuleNotFoundError:
    from ego_networks.src.core import EgoNetwork, NetworkMeasures
    from ego_networks.src.neighborhoods.twitter import TwitterEgoNeighborhood
    from ego_networks.utils.generic import read_data

load_dotenv()
filterwarnings("ignore")
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
INTEGRATED_FOCAL_NODE_ID = os.getenv("INTEGRATED_FOCAL_NODE_ID")
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")


class HomogenousEgoNetwork(EgoNetwork):
    def __init__(
        self,
        focal_node_id: str,
        n_layers: int = 1,
        radius: int = 2,
        nodes=None,
        edges=None,
        storage_bucket: str = None,
    ):
        self._focal_node_id = focal_node_id
        self._n_layers = n_layers
        self._radius = radius
        self._storage_bucket = storage_bucket
        self.G = self.__create_network(nodes, edges)

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def radius(self):
        return self._radius

    @property
    def focal_node_id(self):
        return self._focal_node_id

    def __create_network(self, nodes=None, edges=None):

        if self._storage_bucket is not None:
            edges = read_data(self._storage_bucket, data_type="ties")
            nodes = read_data(
                self._storage_bucket, data_type="node_features"
            ).set_index("id")

        if (edges is None) | (nodes is None):
            raise ValueError(
                "Must provide sample edges and nodes or a storage bucket to read the neighborhood from."
            )

        edges.columns = ["source", "target"]
        edges["source"] = edges["source"].astype(int)
        edges["target"] = edges["target"].astype(int)
        edges["weight"] = 1
        G = nx.from_pandas_edgelist(
            edges, create_using=nx.DiGraph(), edge_attr=True
        )
        for feature in nodes:
            nx.set_node_attributes(
                G,
                pd.Series(nodes[feature]).to_dict(),
                feature,
            )

        G_e = nx.ego_graph(
            G, int(self._focal_node_id), radius=self._radius, undirected=False
        )
        print(
            f"Ego network of layers: {self._n_layers}, radius: {self._radius} created."
        )
        print(f"Nodes: {len(G_e.nodes())}, Edges: {len(G_e.edges())}")
        return G_e

    def create_measures(self, nodes=False, edges=False):
        return EgoNetworkMeasures(self.G, nodes=nodes, edges=edges)


class EgoNetworkMeasures(NetworkMeasures):
    """
    Documentation:
    1. https://faculty.ucr.edu/~hanneman/nettext/C9_Ego_networks.html
    2. networkx: https://networkx.github.io/documentation/stable/reference/algorithms/centrality.html
    3. brokerage: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5325703/

    """

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
        measures["density"] = measures.get("size") / measures.get("pairs")

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
        measures["eigenvector_centrality"] = nx.eigenvector_centrality(self.G)
        measures["pagerank"] = nx.pagerank(self.G)
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
            .set_index("node", append=True)
            .sort_values(ascending=False, by="measure_value")
            .sort_index(level=0, ascending=True)
        )

    def __create_edge_measures(self):
        measures = {}
        return pd.DataFrame.from_dict(
            measures, orient="index", columns=["measure_value"]
        ).rename_axis(index="measure_name")


def main():
    twitter_hood = TwitterEgoNeighborhood(
        focal_node=TWITTER_USERNAME,
        max_radius=2,
        api_bearer_token=TWITTER_API_BEARER_TOKEN,
        storage_bucket=CLOUD_STORAGE_BUCKET,
    )
    twitter_hood.update_neighborhood()
    network = HomogenousEgoNetwork(
        focal_node_id=INTEGRATED_FOCAL_NODE_ID,
        radius=1,
        storage_bucket=CLOUD_STORAGE_BUCKET,
    )
    measures = network.create_measures(nodes=True)
    print(measures.summary_measures)
    print(measures.node_measures)


if __name__ == "__main__":
    main()
