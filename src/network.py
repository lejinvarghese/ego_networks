# -*- coding: utf-8 -*-
"""
A multi layered complex network spanning social media neighborhoods.
"""

from warnings import filterwarnings

import networkx as nx
import pandas as pd

try:
    from src.core import EgoNetwork
    from src.measures import EgoNetworkMeasures
    from utils.generic import read_data
except ModuleNotFoundError:
    from ego_networks.src.core import EgoNetwork
    from ego_networks.src.measures import EgoNetworkMeasures
    from ego_networks.utils.generic import read_data

filterwarnings("ignore")


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
