# -*- coding: utf-8 -*-
"""
Uses Twitter API v2.0 to extract out step neighbors of the focal node
"""


import os
from dotenv import load_dotenv
import time
import ast
from datetime import datetime

import pandas as pd
import dask.dataframe as dd
import tweepy

try:
    from src.network import EgoNetwork
except:
    from ego_networks.src.network import EgoNetwork

load_dotenv()
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")

run_time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")


class TwitterEgoNetwork(EgoNetwork):
    def __init__(self, focal_node: str, max_radius: int, client=None):
        self._focal_node = focal_node
        self._max_radius = int(max_radius)
        self._client = client

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

    def retrieve_edges(self):
        """
        TODO: Convert to recursive calls
        TODO: Initialize with previous neighborhood
        TODO: Evaluate full import refresh frequency vs incremental refresh
        """
        self._focal_node_id = self._retrieve_node_features(
            user_fields=["id"], user_names=[self._focal_node]
        )[0].id

        print(
            f"Retrieving the ego network for {self._focal_node_id}, @max radius: {self._max_radius}"
        )

        previous_edges = dd.read_csv(
            f"{CLOUD_STORAGE_BUCKET}/data/users_following*.csv"
        ).compute()
        previous_edges.following = previous_edges.following.apply(
            ast.literal_eval
        )
        previous_edges = previous_edges.explode("following")

        previous_neighbors_r1 = list(previous_edges.user.unique())
        previous_neighbors_r2 = list(previous_edges.following.unique())

        previous_neighbors_r2 = list(
            set(previous_neighbors_r2) - set(previous_neighbors_r1)
        )
        print(
            f"Previous neighbors \n@radius 1: {len(previous_neighbors_r1)} \n@radius 2: {len(previous_neighbors_r2)} \nPrevious connections: {previous_edges.shape[0]}"
        )

        current_neighbors_r1 = self._retrieve_node_out_neighbors(
            user_id=self._focal_node_id
        ).get("following")

        print(f"Current neighbors \n@radius 1: {len(current_neighbors_r1)}")

        new_neighbors_r1 = list(
            set(current_neighbors_r1) - set(previous_neighbors_r1)
        )

        print(f"New neighbors \n@radius 1: {len(new_neighbors_r1)}")

        new_edges = []
        for u_id in new_neighbors_r1:
            u_data = self._retrieve_node_out_neighbors(user_id=u_id)
            new_edges.append(u_data)

        new_edges = pd.json_normalize(new_edges)

        print(f"Writing new connections: {new_edges.shape}")
        new_edges.to_csv(
            f"{CLOUD_STORAGE_BUCKET}/data/users_following_{run_time}.csv",
            index=False,
        )

        pass

    def retrieve_nodes(self):
        # _focal_node_user_id = self._retrieve_node_user_ids()
        # return _focal_node_user_id
        pass

    def create_network(self):
        pass

    def __copy__(self):
        return TwitterEgoNetwork(
            self._focal_node, self._max_radius, self._client
        )

    def authenticate(self, api_bearer_token):
        client = tweepy.Client(api_bearer_token, wait_on_rate_limit=True)
        self._client = client
        return TwitterEgoNetwork.__copy__(self)

    def _retrieve_node_features(
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

    def _retrieve_node_out_neighbors(
        self, user_id, max_results=1000, total_limit=5000, sleep_timer=0.1
    ):
        following = []
        for o_n in tweepy.Paginator(
            self.client.get_users_following, id=user_id, max_results=max_results
        ).flatten(limit=total_limit):
            time.sleep(sleep_timer)
            following.append(o_n.id)
        print(f"User: {user_id}, Following: {len(following)}")
        return {"user": user_id, "following": following}


def main():
    tn = TwitterEgoNetwork(focal_node=TWITTER_USERNAME, max_radius=2)
    tn_a = tn.authenticate(api_bearer_token=TWITTER_API_BEARER_TOKEN)

    tn_a.retrieve_edges()


if __name__ == "__main__":
    main()
