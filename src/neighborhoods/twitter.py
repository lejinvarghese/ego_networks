# -*- coding: utf-8 -*-
"""
Uses Twitter API v2.0 to extract out step neighbors of the focal node
"""


import time
from warnings import filterwarnings

import pandas as pd
from dotenv import load_dotenv

try:
    from src.core import EgoNeighborhood
    from utils.api.twitter import authenticate, get_users, get_users_following
    from utils.default import split_into_batches
    from utils.io import DataReader, DataWriter
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.src.core import EgoNeighborhood
    from ego_networks.utils.api.twitter import (
        authenticate,
        get_users,
        get_users_following,
    )
    from ego_networks.utils.default import split_into_batches
    from ego_networks.utils.io import DataReader, DataWriter
    from ego_networks.utils.custom_logger import CustomLogger

load_dotenv()
filterwarnings("ignore")
logger = CustomLogger(__name__)


class TwitterEgoNeighborhood(EgoNeighborhood):
    def __init__(
        self,
        focal_node: str,
        max_radius: int,
        api_bearer_token: str = None,
    ):
        self._layer = "twitter"
        self._focal_node = focal_node
        self._max_radius = int(max_radius)
        self._client = authenticate(api_bearer_token)
        self._previous_ties = DataReader(data_type="ties").run()
        self._previous_node_features = DataReader(
            data_type="node_features"
        ).run()
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

    def update_neighborhood(self, mode="append"):
        """
        Updates the neighborhood of the focal node
        """

        if mode == "append":
            logger.info(
                f"Appending to the ego neigborhood for {self._focal_node_id}, @max radius: {self._max_radius}"
            )
            new_ties, nodes = self.update_ties()
            if new_ties.shape[0] > 0:
                writer = DataWriter(data=new_ties, data_type="ties")
                writer.run(append=True)
            else:
                logger.info("No new ties to append to the neighborhood")

            new_node_features = self.update_node_features(nodes=nodes)
            if new_node_features.shape[0] > 0:
                writer = DataWriter(
                    data=new_node_features, data_type="node_features"
                )
                writer.run(append=True)
            else:
                logger.info(
                    "No new node features to append to the neighborhood"
                )
        elif mode == "delete":
            logger.warning(
                f"Deleting stale ties in the ego neigborhood for {self._focal_node_id}, @max radius: {self._max_radius}"
            )

            cleansed_ties, cleansed_node_features = self.delete_ties()
            if cleansed_ties.shape[0] > 0:
                writer = DataWriter(data=cleansed_ties, data_type="ties")
                writer.run(append=False)
            else:
                logger.info("No new ties to delete from the neighborhood")

            if cleansed_node_features.shape[0] > 0:
                writer = DataWriter(
                    data=cleansed_node_features, data_type="node_features"
                )
                writer.run(append=False)
            else:
                logger.info(
                    "No new node features to delete from the neighborhood"
                )

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

        alters[1]["current"] = set(
            get_users_following(
                client=self._client, user_id=self._focal_node_id
            ).get("following")
        )

        alters[1]["new"] = alters.get(1).get("current") - alters.get(1).get(
            "previous"
        )

        logger.info(
            f"Previous alters \n@radius 1: {len(alters.get(1).get('previous'))} \n@radius 2: {len(alters.get(2).get('previous'))} \nPrevious ties: {self.previous_ties.shape[0]}"
        )
        logger.info(
            f"Current alters \n@radius 1: {len(alters.get(1).get('current'))}"
        )
        logger.info(f"New alters \n@radius 1: {len(alters.get(1).get('new'))}")

        # get new ties, or new edges
        new_ties = [
            {
                "user": self._focal_node_id,
                "following": alters.get(1).get("new"),
            }
        ]

        for u_id in alters.get(1).get("new"):
            u_data = get_users_following(client=self._client, user_id=u_id)
            if len(u_data.get("following")) > 0:
                new_ties.append(u_data)
                alters[2]["new"].update(set(u_data.get("following")))

        new_ties = pd.json_normalize(new_ties)

        # recalculate all new nodes
        alters_all = set()
        for i in range(1, self._max_radius + 1):
            alters_all.update(alters.get(i).get("previous"))
            alters_all.update(alters.get(i).get("current"))

        # remove nan values
        alters_all = {x for x in alters_all if x == x}
        return new_ties, alters_all

    def update_tie_features(self):
        pass

    def update_node_features(self, nodes, sleep_time=0.1):
        try:
            previous_nodes_with_features = set(
                self.previous_node_features.index.unique()
            )

            logger.info(
                f"Previous nodes with features: {len(previous_nodes_with_features)}"
            )
        except AttributeError:
            previous_nodes_with_features = set()

        new_nodes = set(nodes - previous_nodes_with_features)
        new_nodes.add(self._focal_node_id)

        logger.info(
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
            "public_metrics",
            "username",
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
        return new_node_features

    def delete_ties(self):

        previous_alters = set(self.previous_ties.user.unique())
        previous_alters.remove(self._focal_node_id)

        current_alters = set(
            get_users_following(
                client=self._client, user_id=self._focal_node_id
            ).get("following")
        )

        removed_alters = previous_alters - current_alters

        # remove ties
        cleansed_ties = self._previous_ties[
            ~(self._previous_ties.user.isin(removed_alters))
        ]
        cleansed_ties = (
            cleansed_ties.groupby("user")["following"]
            .apply(list)
            .reset_index(name="following")
        )
        cleansed_node_features = self._previous_node_features[
            ~(
                self._previous_node_features.index.isin(
                    removed_alters
                )
            )
        ]
        cleansed_node_features = cleansed_node_features.reset_index()

        logger.info(f"Previous Ties: {self._previous_ties.shape}")
        logger.info(
            f"Previous Node Features: {self._previous_node_features.shape}"
        )
        logger.warning(f"Removed alters: {len(removed_alters)}")
        logger.warning(f"Cleansed Ties: {cleansed_ties.shape}")
        logger.warning(
            f"Cleansed Node Features: {cleansed_node_features.shape}"
        )
        return cleansed_ties, cleansed_node_features