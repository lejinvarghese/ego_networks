"""
Uses Twitter API v2.0 to extract out step neighbors of the focal node
"""

import os
from abc import ABC, abstractmethod, abstractproperty
import tweepy
from dotenv import load_dotenv
import time
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import pandas as pd
import dask.dataframe as dd
import tweepy

load_dotenv()
TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")

n_threads = cpu_count() - 1
run_time = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")


class EgoNetwork(ABC):
    @abstractproperty
    def focal_node(self):
        pass

    @abstractproperty
    def radius(self):
        pass

    @abstractmethod
    def retrieve_edges(self):
        pass

    @abstractmethod
    def retrieve_node_features(self, node):
        pass

    @abstractmethod
    def create_network(self):
        pass


class TwitterEgoNetwork(EgoNetwork):
    def __init__(self, focal_node: str, radius: int, client=None):
        self._focal_node = focal_node
        self._radius = int(radius)
        self._client = client

    @property
    def focal_node(self):
        return self._focal_node

    @property
    def radius(self):
        return self._radius

    @property
    def client(self):
        return self._client

    @client.setter
    def client(self, value):
        self._client = value

    def __copy__(self):
        return TwitterEgoNetwork(self._focal_node, self._radius, self._client)

    def authenticate(self, api_bearer_token):
        client = tweepy.Client(api_bearer_token, wait_on_rate_limit=True)
        self._client = client
        return TwitterEgoNetwork.__copy__(self)

    def retrieve_edges(self):
        existing_users = list(
            dd.read_csv(f"{CLOUD_STORAGE_BUCKET}/data/users_following*.csv")
            .compute()
            .user.unique()
        )
        print(f"Previously following: {len(existing_users)}")

    def retrieve_node_features(self, node):
        # _focal_node_user_id = self._retrive_user_id()
        # return _focal_node_user_id
        pass

    def _retrive_user_id(self):
        return self.client.get_user(
            username=self._focal_node,
            user_fields=["id"],
        ).data.id

    def create_network(self):
        pass


def main():
    tn = TwitterEgoNetwork(focal_node=TWITTER_USERNAME, radius=2)
    tn_a = tn.authenticate(api_bearer_token=TWITTER_API_BEARER_TOKEN)
    print(f"Retrieving the Ego Network for {tn_a.focal_node}, of Radius: {tn_a.radius}")

    tn_a.retrieve_edges()


if __name__ == "__main__":
    main()
