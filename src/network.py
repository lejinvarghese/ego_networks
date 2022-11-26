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
    from utils.io import DataReader, DataWriter
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.src.core import EgoNetwork
    from ego_networks.src.measures import EgoNetworkMeasures
    from ego_networks.utils.io import DataReader, DataWriter
    from ego_networks.utils.custom_logger import CustomLogger

filterwarnings("ignore")
logger = CustomLogger(__name__)


class HomogenousEgoNetwork(EgoNetwork):
    def __init__(
        self,
        focal_node_id: str,
        n_layers: int = 1,
        radius: int = 2,
        nodes=None,
        edges=None,
        use_cache: bool = True,
    ):
        self._focal_node_id = focal_node_id
        self._n_layers = n_layers
        self._radius = radius
        self._use_cache = use_cache
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

        if self._use_cache:
            edges = DataReader(data_type="ties").run()
            nodes = DataReader(data_type="node_features").run()

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
        logger.info(
            f"Ego network of layers: {self._n_layers}, radius: {self._radius} created."
        )
        logger.info(
            f"Nodes: {G_e.number_of_nodes()}, Edges: {G_e.number_of_edges()}"
        )
        return G_e

    def create_measures(
        self, calculate_nodes=False, calculate_edges=False, cache=True
    ):
        measures = EgoNetworkMeasures(
            self.G,
            calculate_nodes=calculate_nodes,
            calculate_edges=calculate_edges,
        )
        if cache:
            writer = DataWriter(data=measures.node_measures, data_type="node_measures")
            writer.run(append=False)
        return measures

    def get_ego_graph_at_radius(self, radius: int = 1):
        return nx.ego_graph(self.G, int(self._focal_node_id), radius=radius)

    def get_ego_user_attributes(
        self, radius: int = 1, attribute: str = "username"
    ):
        g_1 = self.get_ego_graph_at_radius(radius)
        return nx.get_node_attributes(g_1, attribute)
