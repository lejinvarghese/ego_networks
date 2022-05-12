# -*- coding: utf-8 -*-
"""
Uses Twitter API v2.0 to extract out step neighbors of the focal node
"""


import os
import time
from warnings import filterwarnings

import networkx as nx
import pandas as pd
from dotenv import load_dotenv

try:
    from src.core import EgoNeighborhood, EgoNetwork
    from utils.api.twitter import authenticate, get_users, get_users_following
    from utils.generic import read_data, split_into_batches, write_data
except ModuleNotFoundError:
    from ego_networks.src.core import EgoNeighborhood, EgoNetwork
    from ego_networks.utils.api.twitter import (
        authenticate,
        get_users,
        get_users_following,
    )
    from ego_networks.utils.generic import (
        read_data,
        split_into_batches,
        write_data,
    )

load_dotenv()
filterwarnings("ignore")
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
INTEGRATED_FOCAL_NODE_ID = os.getenv("INTEGRATED_FOCAL_NODE_ID")
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")


class TwitterEgoNeighborhood(EgoNeighborhood):
    def __init__(
        self,
        focal_node: str,
        max_radius: int,
        api_bearer_token: str = None,
        storage_bucket: str = None,
    ):
        self._layer = "twitter"
        self._focal_node = focal_node
        self._max_radius = int(max_radius)
        self._storage_bucket = storage_bucket
        self._client = authenticate(api_bearer_token)
        self._previous_ties = read_data(self._storage_bucket, data_type="ties")
        self._previous_node_features = read_data(
            self._storage_bucket, data_type="node_features"
        )
        self._focal_node_id = get_users(
            client=self._client,
            user_fields=["id"],
            user_names=[self._focal_node],
        )[0].id

    @property
    def layer(self):
        return self._layer

    @property
    def focal_node(self):
        return self._focal_node

    @property
    def max_radius(self):
        return self._max_radius

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    @property
    def previous_ties(self):
        return self._previous_ties

    @property
    def previous_node_features(self):
        return self._previous_node_features

    @property
    def focal_node_id(self):
        return self._focal_node_id

    def update_neighborhood(self):
        """
        Updates the neighborhood of the focal node
        """

        print(
            f"Updating the ego neigborhood for {self._focal_node_id}, @max radius: {self._max_radius}"
        )

        new_ties, nodes = self.update_ties()
        new_node_features = self.update_node_features(nodes=nodes)

        if new_node_features.shape[0] > 0:
            write_data(
                self._storage_bucket,
                new_node_features,
                data_type="node_features",
            )
        else:
            print("No new node features to update neighborhood")

        if new_ties.shape[0] > 0:
            write_data(
                self._storage_bucket,
                new_ties,
                data_type="ties",
            )
        else:
            print("No new ties to update neighborhood")

    def update_ties(self):

        alters = {}
        alters = {
            i: {"previous": set(), "current": set(), "new": set()}
            for i in range(1, self._max_radius + 1)
        }
        alters[1]["previous"] = set(self.previous_ties.user.unique())
        alters[2]["previous"] = (
            set(self.previous_ties.following.unique()) - alters[1]["previous"]
        )

        print(
            f"Previous alters \n@radius 1: {len(alters.get(1).get('previous'))} \n@radius 2: {len(alters.get(2).get('previous'))} \nPrevious ties: {self.previous_ties.shape[0]}"
        )

        alters[1]["current"] = set(
            get_users_following(
                client=self._client, user_id=self._focal_node_id
            ).get("following")
        )

        print(
            f"Current alters \n@radius 1: {len(alters.get(1).get('current'))}"
        )

        alters[1]["new"] = alters.get(1).get("current") - alters.get(1).get(
            "previous"
        )

        print(f"New alters \n@radius 1: {len(alters.get(1).get('new'))}")

        new_ties = [
            {
                "user": self._focal_node_id,
                "following": alters.get(1).get("current"),
            }
        ]

        for u_id in alters.get(1).get("new"):
            u_data = get_users_following(client=self._client, user_id=u_id)
            if len(u_data.get("following")) > 0:
                new_ties.append(u_data)
                alters[2]["new"].update(set(u_data.get("following")))

        new_ties = pd.json_normalize(new_ties)

        total_alters = set()
        for i in range(1, self._max_radius + 1):
            total_alters.update(alters.get(i).get("previous"))
            total_alters.update(alters.get(i).get("current"))

        # remove nan values
        total_alters = {x for x in total_alters if x == x}
        return new_ties, total_alters

    def update_tie_features(self):
        pass

    def update_node_features(self, nodes, sleep_time=0.1):
        try:
            previous_nodes_with_features = set(
                self.previous_node_features.id.unique()
            )

            print(
                f"Previous nodes with features: {len(previous_nodes_with_features)}"
            )
        except AttributeError:
            previous_nodes_with_features = set()

        new_nodes = set(nodes - previous_nodes_with_features)
        new_nodes.add(self._focal_node_id)

        print(
            f"All nodes within radius {self._max_radius}: {len(nodes)}, \nNew nodes: {len(new_nodes)}"
        )

        max_api_batch_size = 100
        if len(new_nodes) > max_api_batch_size:
            new_node_batches = split_into_batches(
                list(new_nodes), batch_size=max_api_batch_size
            )
        else:
            new_node_batches = [list(new_nodes)]

        new_node_features = []
        feature_fields = [
            "profile_image_url",
            "username",
            "public_metrics",
            "verified",
        ]
        for batch in new_node_batches:
            time.sleep(sleep_time)
            new_node_features.append(
                pd.DataFrame(
                    get_users(
                        client=self._client,
                        user_fields=feature_fields,
                        user_ids=batch,
                    )
                )
            )

        new_node_features = pd.concat(new_node_features)
        new_node_features["withheld"] = None
        return new_node_features


class HomogenousEgoNetwork(EgoNetwork):
    def __init__(
        self,
        focal_node_id: str,
        n_layers: int = 1,
        radius: int = 2,
        storage_bucket: str = None,
    ):
        self._focal_node_id = focal_node_id
        self._n_layers = n_layers
        self._radius = radius
        self._storage_bucket = storage_bucket
        self.G = self.__create_network()

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def radius(self):
        return self._radius

    @property
    def focal_node_id(self):
        return self._focal_node_id

    def __create_network(self, edges=None, nodes=None):

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

    class Measures:
        def __init__(self, G, nodes=False, edges=False):
            self.G = G
            self.summary_measures = self.__create_graph_measures()
            if nodes:
                self.node_measures = self.__create_nodes_measures()
            if edges:
                self.edge_measures = self.__create_edges_measures()

        def __create_graph_measures(self):
            measures = {}
            measures["n_nodes"] = measures["size"] = len(self.G.nodes())
            measures["n_edges"] = measures["ties"] = len(self.G.edges())
            measures["pairs"] = measures.get("size") * (
                measures.get("size") - 1
            )
            measures["density"] = round(
                measures.get("size") / measures.get("pairs"), 2
            )
            measures["transitivity"] = round(nx.transitivity(self.G), 2)
            measures["average_clustering"] = round(
                nx.average_clustering(self.G), 2
            )
            measures["n_strongly_connected_components"] = round(
                nx.number_strongly_connected_components(self.G)
            )
            measures["n_attracting_components"] = round(
                nx.number_attracting_components(self.G)
            )
            measures["global_reaching_centrality"] = round(
                nx.global_reaching_centrality(self.G)
            )
            return (
                pd.DataFrame.from_dict(
                    measures, orient="index", columns=["measure_value"]
                )
                .rename_axis(index="measure_name")
                .sort_index()
            )

        def __create_nodes_measures(self):
            measures = {}
            measures["degree_centrality"] = nx.degree_centrality(self.G)
            measures["eigenvector_centrality"] = nx.eigenvector_centrality(
                self.G
            )
            return (
                pd.DataFrame.from_dict(measures, orient="index")
                .rename_axis(index="measure_name")
                .melt(
                    ignore_index=False,
                    var_name="node",
                    value_name="measure_value",
                )
                .round(2)
                .set_index("node", append=True)
                .sort_index()
            )

        def __create_edges_measures(self):
            measures = {}
            return pd.DataFrame.from_dict(
                measures, orient="index", columns=["measure_value"]
            ).rename_axis(index="measure_name")

    def create_measures(self, nodes=False, edges=False):
        return self.Measures(self.G, nodes=nodes, edges=edges)


def main():
    # twitter_hood = TwitterEgoNeighborhood(
    #     focal_node=TWITTER_USERNAME,
    #     max_radius=2,
    #     api_bearer_token=TWITTER_API_BEARER_TOKEN,
    #     storage_bucket=CLOUD_STORAGE_BUCKET,
    # )
    # twitter_hood.update_neighborhood()
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
