# -*- coding: utf-8 -*-
"""
Uses Twitter API v2.0 to extract out step neighbors of the focal node
"""


import ast
import os
import time
from datetime import datetime

import dask.dataframe as dd
import pandas as pd
import tweepy
from dotenv import load_dotenv

try:
    from src.network import EgoNetwork
except ModuleNotFoundError:
    from ego_networks.src.network import EgoNetwork

load_dotenv()
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
        self._client = self.__authenticate(api_bearer_token)
        self._previous_ties = self.__retrieve_previous_ties(storage_bucket)

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

    def create_neighborhood(self):
        """
        TODO: Convert to recursive calls
        TODO: Initialize with previous neighborhood
        TODO: Evaluate full import refresh frequency vs incremental refresh
        """
        self._focal_node_id = self.retrieve_node_features(
            user_fields=["id"], user_names=[self._focal_node]
        )[0].id

        print(
            f"Retrieving the ego network for {self._focal_node_id}, @max radius: {self._max_radius}"
        )

        previous_alters_r1 = list(self.previous_ties.user.unique())
        previous_alters_r2 = list(self.previous_ties.following.unique())

        previous_alters_r2 = list(set(previous_alters_r2) - set(previous_alters_r1))
        print(
            f"Previous neighbors \n@radius 1: {len(previous_alters_r1)} \n@radius 2: {len(previous_alters_r2)} \nPrevious connections: {self.previous_ties.shape[0]}"
        )

        current_alters_r1 = self.retrieve_ties(user_id=self._focal_node_id).get(
            "following"
        )

        print(f"Current neighbors \n@radius 1: {len(current_alters_r1)}")

        new_alters_r1 = list(set(current_alters_r1) - set(previous_alters_r1))

        print(f"New neighbors \n@radius 1: {len(new_alters_r1)}")

        new_ties = []
        for u_id in new_alters_r1:
            u_data = self.retrieve_ties(user_id=u_id)
            new_ties.append(u_data)

        new_ties = pd.json_normalize(new_ties)

        print(f"Writing new connections: {new_ties.shape}")
        new_ties.to_csv(
            f"{CLOUD_STORAGE_BUCKET}/data/users_following_{run_time}.csv",
            index=False,
        )

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

    def retrieve_node_features(self, user_fields, user_names=None, user_ids=None):

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
            raise ValueError("Either one of user_names or user_ids should be provided")

    def __authenticate(self, api_bearer_token):
        client = tweepy.Client(api_bearer_token, wait_on_rate_limit=True)
        return client

    def __retrieve_previous_ties(self, storage_bucket):
        try:
            previous_ties = dd.read_csv(
                f"{storage_bucket}/data/users_following*.csv"
            ).compute()
            print(f"Storage bucket authenticated")
            previous_ties.following = previous_ties.following.apply(ast.literal_eval)
            previous_ties = previous_ties.explode("following")
        except Exception as error:
            print(f"Storage bucket not found, {error}")
            previous_ties = pd.DataFrame()

        return previous_ties


def main():
    ego_network = TwitterEgoNetwork(
        focal_node=TWITTER_USERNAME,
        max_radius=2,
        api_bearer_token=TWITTER_API_BEARER_TOKEN,
        storage_bucket=CLOUD_STORAGE_BUCKET,
    )
    ego_network.create_neighborhood()


if __name__ == "__main__":
    main()
