# -*- coding: utf-8 -*-
"""
Uses Twitter API v2.0 to extract out step neighbors of the focal node
"""


import ast
import os
import time
from datetime import datetime
from warnings import filterwarnings

import dask.dataframe as dd
import networkx as nx
import numpy as np
import pandas as pd
import tweepy
from dotenv import load_dotenv

try:
    from src.network import EgoNetwork
except ModuleNotFoundError:
    from ego_networks.src.network import EgoNetwork

load_dotenv()
filterwarnings("ignore")
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")
run_time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")


class TwitterEgoNetwork(EgoNetwork):
    def __init__(
        self,
        focal_node: str,
        max_radius: int,
        api_bearer_token=None,
        storage_bucket=None,
    ):
        self._focal_node = focal_node
        self._max_radius = int(max_radius)
        self._api_bearer_token = api_bearer_token
        self._storage_bucket = storage_bucket
        self._client = self.__authenticate()
        self._previous_ties = self.__retrieve_previous_ties()
        self._previous_alter_features = (
            self.__retrieve_previous_alter_features()
        )
        self._focal_node_id = self.retrieve_node_features(
            user_fields=["id"], user_names=[self._focal_node]
        )[0].id

    @property
    def focal_node(self):
        return self._focal_node

    @property
    def max_radius(self):
        return self._max_radius

    @property
    def api_bearer_token(self):
        return self._api_bearer_token

    @property
    def storage_bucket(self):
        return self._storage_bucket

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
    def previous_alter_features(self):
        return self._previous_alter_features

    @property
    def focal_node_id(self):
        return self._focal_node_id

    def create_network(self):
        edges = self._previous_ties.copy()
        nodes = self._previous_alter_features.copy().set_index("id")
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

        print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
        print(f"Nodes with Features: {nodes.shape[0]}")
        return G

    def update_neighborhood(self):
        """
        Updates the neighborhood of the focal node
        """

        print(
            f"Updating the ego neigborhood for {self._focal_node_id}, @max radius: {self._max_radius}"
        )

        _new_ties, _all_alters = self.update_ties()
        _new_alter_features = self.update_alter_features(_all_alters)

        if _new_alter_features.shape[0] > 0:
            print(f"Writing new alter features: {_new_alter_features.shape}")
            _new_alter_features.to_csv(
                f"{self._storage_bucket}/data/node_features_{run_time}.csv",
                index=False,
            )
        else:
            print(f"No new alter features to update neighborhood")

        if _new_ties.shape[0] > 0:
            print(f"Writing new ties: {_new_ties.shape}")
            _new_ties.to_csv(
                f"{self._storage_bucket}/data/users_following_{run_time}.csv",
                index=False,
            )
        else:
            print(f"No new ties to update neighborhood")

    def update_ties(self):

        previous_alters_r1 = list(self.previous_ties.user.unique())
        previous_alters_r2 = list(self.previous_ties.following.unique())

        previous_alters_r2 = list(
            set(previous_alters_r2) - set(previous_alters_r1)
        )
        print(
            f"Previous alters \n@radius 1: {len(previous_alters_r1)} \n@radius 2: {len(previous_alters_r2)} \nPrevious ties: {self.previous_ties.shape[0]}"
        )

        current_alters_r1 = self.retrieve_ties(user_id=self._focal_node_id).get(
            "following"
        )

        print(f"Current alters \n@radius 1: {len(current_alters_r1)}")

        new_alters_r1 = list(set(current_alters_r1) - set(previous_alters_r1))

        print(f"New alters \n@radius 1: {len(new_alters_r1)}")

        new_ties, new_alters_r2 = [], []
        for u_id in new_alters_r1:
            u_data = self.retrieve_ties(user_id=u_id)
            if len(u_data.get("following")) > 0:
                new_ties.append(u_data)
                new_alters_r2.append(u_data.get("following"))

        new_ties = pd.json_normalize(new_ties)
        new_ties = pd.concat(
            [
                new_ties,
                pd.DataFrame(
                    {
                        "user": self._focal_node_id,
                        "following": [current_alters_r1],
                    }
                ),
            ]
        )
        new_alters_r2 = set(
            [item for sub_list in new_alters_r2 for item in sub_list]
        )

        _all_alters = np.array(
            list(
                (
                    set(new_alters_r1)
                    | set(new_alters_r2)
                    | set(previous_alters_r1)
                    | set(previous_alters_r2)
                )
            )
        )
        return new_ties, _all_alters[~np.isnan(_all_alters)].astype(int)

    def update_alter_features(self, alters):
        try:
            previous_alters_with_features = list(
                self.previous_alter_features.id.unique()
            )

            print(
                f"Previous alters with features: {len(previous_alters_with_features)}"
            )
        except AttributeError:
            previous_alters_with_features = []

        new_alters = list(set(alters) - set(previous_alters_with_features)) + [
            self._focal_node_id
        ]

        print(
            f"All alters within radius {self._max_radius}: {len(alters)}, \nNew alters: {len(new_alters)}"
        )

        max_api_batch_size = 100
        if len(new_alters) > max_api_batch_size:
            new_alter_batches = self.__get_batches(
                new_alters, batch_size=max_api_batch_size
            )
        else:
            new_alter_batches = [new_alters]

        new_alter_features = []
        alter_feature_fields = [
            "profile_image_url",
            "username",
            "public_metrics",
            "verified",
        ]
        for batch in new_alter_batches:
            time.sleep(0.1)
            new_alter_features.append(
                pd.DataFrame(
                    self.retrieve_node_features(
                        user_fields=alter_feature_fields,
                        user_ids=batch,
                    )
                )
            )

        new_alter_features = pd.concat(new_alter_features)
        new_alter_features["withheld"] = None
        return new_alter_features

    def retrieve_ties(
        self, user_id, max_results=1000, total_limit=5000, sleep_timer=0.1
    ):
        following = []
        for neighbor in tweepy.Paginator(
            self.client.get_users_following, id=user_id, max_results=max_results
        ).flatten(limit=total_limit):
            time.sleep(sleep_timer)
            following.append(neighbor.id)
        print(f"User: {user_id}, Following: {len(following)}")
        return {"user": user_id, "following": following}

    def retrieve_tie_features(self):
        pass

    def retrieve_node_features(
        self, user_fields, user_names=None, user_ids=None
    ):

        if user_ids:
            return self.client.get_users(
                ids=user_ids,
                user_fields=user_fields,
            ).data
        elif user_names:
            return self.client.get_users(
                usernames=user_names,
                user_fields=user_fields,
            ).data
        else:
            raise ValueError(
                "Either one of user_names or user_ids should be provided"
            )

    def __authenticate(self):
        client = tweepy.Client(self._api_bearer_token, wait_on_rate_limit=True)
        return client

    def __retrieve_previous_ties(self):
        try:
            previous_ties = dd.read_csv(
                f"{self._storage_bucket}/data/users_following*.csv"
            ).compute()
            print(f"Storage bucket authenticated")
            previous_ties.following = previous_ties.following.apply(
                ast.literal_eval
            )
            previous_ties = previous_ties.explode("following")
        except Exception as error:
            print(f"Storage bucket not found, {error}")
            previous_ties = pd.DataFrame()

        return previous_ties.dropna().drop_duplicates()

    def __retrieve_previous_alter_features(self):
        try:
            previous_alter_features = dd.read_csv(
                f"{self._storage_bucket}/data/node_features*.csv",
                dtype={"withheld": "object"},
            ).compute()
            print(f"Storage bucket authenticated")
        except Exception as error:
            print(f"Storage bucket not found, {error}")
            previous_alter_features = pd.DataFrame()

        return previous_alter_features.drop(
            columns="withheld"
        ).drop_duplicates()

    def __get_batches(self, src_list: list, batch_size: int):
        batches = np.array_split(src_list, len(src_list) // (batch_size - 1))
        batches = [batch.tolist() for batch in batches]

        print(
            f"Total batches: {np.shape(batches)[0]}, batch size:{np.shape(batches[0])[0]}"
        )
        return batches


def main():
    ego_network = TwitterEgoNetwork(
        focal_node=TWITTER_USERNAME,
        max_radius=2,
        api_bearer_token=TWITTER_API_BEARER_TOKEN,
        storage_bucket=CLOUD_STORAGE_BUCKET,
    )
    ego_network.update_neighborhood()
    G = ego_network.create_network()


if __name__ == "__main__":
    main()
