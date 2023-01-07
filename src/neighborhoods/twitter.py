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
    from utils.api.twitter import (
        authenticate,
        get_users,
        get_users_following,
        get_engagement,
    )
    from utils.default import split_into_batches
    from utils.io import DataReader, DataWriter
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.src.core import EgoNeighborhood
    from ego_networks.utils.api.twitter import (
        authenticate,
        get_users,
        get_users_following,
        get_engagement,
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
        self.alters = self.__instantiate_alter_state()

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
            logger.debug(
                f"Appending to the ego neigborhood for {self.focal_node_id}, @max radius: {self.max_radius}"
            )
            new_ties, nodes = self.update_ties()
            if new_ties.shape[0] > 0:
                writer = DataWriter(data=new_ties, data_type="ties")
                writer.run(append=True)
            else:
                logger.debug("No new ties to append to the neighborhood")

            new_node_features = self.update_node_features(nodes=nodes)
            if new_node_features.shape[0] > 0:
                writer = DataWriter(
                    data=new_node_features, data_type="node_features"
                )
                writer.run(append=True)
            else:
                logger.debug(
                    "No new node features to append to the neighborhood"
                )

            tie_features = self.update_tie_features(sleep_time=10)
            if tie_features.shape[0] > 0:
                writer = DataWriter(data=tie_features, data_type="tie_features")
                writer.run(append=True)
            else:
                logger.debug(
                    "No new tie features to append to the neighborhood"
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
                logger.debug("No new ties to delete from the neighborhood")

            if cleansed_node_features.shape[0] > 0:
                writer = DataWriter(
                    data=cleansed_node_features, data_type="node_features"
                )
                writer.run(append=False)
            else:
                logger.debug(
                    "No new node features to delete from the neighborhood"
                )

    def update_ties(self):
        logger.debug(f"Updating ties")

        # get new ties, or new edges
        new_ties = [
            {
                "user": self.focal_node_id,
                "following": self.alters.get(1).get("new"),
            }
        ]

        if self._max_radius == 2:
            for u_id in self.alters.get(1).get("new"):
                u_data = get_users_following(client=self.client, user_id=u_id)
                if len(u_data.get("following")) > 0:
                    new_ties.append(u_data)
                    self.alters[2]["new"].update(set(u_data.get("following")))

        new_ties = pd.json_normalize(new_ties)

        # recalculate all new nodes
        alters_all = set()
        for i in range(1, self.max_radius + 1):
            alters_all.update(self.alters.get(i).get("previous"))
            alters_all.update(self.alters.get(i).get("current"))

        # remove nan values
        alters_all = {x for x in alters_all if x == x}
        return new_ties, alters_all

    def update_tie_features(self, max_users=None, sleep_time=0.1):
        logger.debug(f"Updating tie features")
        tie_features = pd.DataFrame()
        content_types = ["tweets", "mentions", "likes"]
        user_ids = self.alters.get(1).get("current")
        if max_users:
            user_ids = set(list(user_ids)[:max_users])

        for c in content_types:
            time.sleep(sleep_time)
            for i, user_id in enumerate(user_ids):
                if i % 100 == 0:
                    time.sleep(sleep_time)
                try:
                    data = get_engagement(
                        client=self.client, user_id=user_id, content_type=c
                    )
                    for d in data:
                        tie_features = tie_features.append(
                            {
                                "user_id": user_id,
                                "content": c,
                                "timestamp": d.created_at,
                                "tweet_id": d.id,
                                "public_metrics": d.public_metrics,
                                "in_reply_to_user_id": d.in_reply_to_user_id,
                                "hashtags": [
                                    h.get("tag")
                                    for h in d.get("entities").get(
                                        "hashtags", ()
                                    )
                                ],
                                "context": [
                                    t.get("entity", ()).get("name").lower()
                                    for t in d.context_annotations
                                ],
                            },
                            ignore_index=True,
                        )
                except:
                    continue
        return tie_features

    def update_node_features(self, nodes, sleep_time=0.1):
        logger.debug(f"Updating node features")
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
        new_nodes.add(self.focal_node_id)

        logger.info(
            f"Nodes within radius {self._max_radius}: {len(nodes)}, new nodes: {len(new_nodes)}"
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
                        client=self.client,
                        user_fields=feature_fields,
                        user_ids=batch,
                    )
                )
            )

        new_node_features = pd.concat(new_node_features)
        return new_node_features

    def delete_ties(self):
        logger.warning(f"Deleting ties and node features")

        # remove ties
        cleansed_ties = self.previous_ties[
            ~(self.previous_ties.user.isin(self.alters.get(1).get("removed")))
        ]
        # cleansed_ties = self.previous_ties[
        #     ~(self.previous_ties.following.isin(self.alters.get(1).get("removed")))
        # ]
        cleansed_ties = (
            cleansed_ties.groupby("user")["following"]
            .apply(list)
            .reset_index(name="following")
        )
        cleansed_node_features = self.previous_node_features[
            ~(
                self.previous_node_features.index.isin(
                    self.alters.get(1).get("removed")
                )
            )
        ]
        cleansed_node_features = cleansed_node_features.reset_index()

        logger.info(f"Previous Ties: {self.previous_ties.shape}")
        logger.info(
            f"Previous Node Features: {self.previous_node_features.shape}"
        )
        logger.warning(
            f"Removed alters: {len(self.alters.get(1).get('removed'))}"
        )
        logger.warning(f"Cleansed Ties: {cleansed_ties.shape}")
        logger.warning(
            f"Cleansed Node Features: {cleansed_node_features.shape}"
        )
        return cleansed_ties, cleansed_node_features

    def __instantiate_alter_state(self):
        alters = {
            i: {
                "previous": set(),
                "current": set(),
                "new": set(),
                "removed": set(),
            }
            for i in range(1, self.max_radius + 1)
        }
        alters[1]["previous"] = set(self.previous_ties.user.unique())
        alters[1]["previous"] = (
            alters.get(1).get("previous").difference(set([self.focal_node_id]))
        )
        alters[1]["current"] = set(
            get_users_following(
                client=self.client, user_id=self.focal_node_id
            ).get("following")
        )
        alters[1]["new"] = alters.get(1).get("current") - alters.get(1).get(
            "previous"
        )
        alters[1]["removed"] = alters.get(1).get("previous") - alters.get(
            1
        ).get("current")

        if self.max_radius == 2:
            alters[2]["previous"] = set(
                self.previous_ties.following.unique()
            ) - alters.get(1).get("previous")

        logger.info(
            f"Previous alters @radius 1: {len(alters.get(1).get('previous'))}"
        )
        if self.max_radius == 2:
            logger.info(
                f"Previous alters @radius 2: {len(alters.get(2).get('previous'))}"
            )
        logger.info(
            f"Current alters @radius 1: {len(alters.get(1).get('current'))}"
        )
        logger.info(f"New alters @radius 1: {len(alters.get(1).get('new'))}")
        logger.info(
            f"New alters @radius 1: {len(alters.get(1).get('removed'))}"
        )
        return alters
