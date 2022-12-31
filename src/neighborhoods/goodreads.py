# -*- coding: utf-8 -*-
"""
Uses a Goodreads scraper to extract out step neighbors of the focal node
"""
from warnings import filterwarnings
import pandas as pd
from dotenv import load_dotenv

try:
    from src.core import EgoNeighborhood
    from utils.api.goodreads import get_shelf_data
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.src.core import EgoNeighborhood
    from ego_networks.utils.custom_logger import CustomLogger
    from ego_networks.utils.api.goodreads import get_shelf_data

load_dotenv()
filterwarnings("ignore")
logger = CustomLogger(__name__)


class GoodreadsEgoNeighborhood(EgoNeighborhood):
    def __init__(
        self,
        focal_node: str,
        max_radius: int,
    ):
        self._layer = "goodreads"
        self._focal_node = focal_node
        self._max_radius = int(max_radius)
        self._focal_node_id = self._focal_node

    @property
    def layer(self):
        return self._layer

    @property
    def focal_node(self):
        return self._focal_node

    @property
    def max_radius(self):
        return self._max_radius

    def update_neighborhood(self):
        logger.debug(
            f"Retrieving Goodreads neighborhood for: {self._focal_node}"
        )
        node_features = self.update_node_features()
        ties = self.update_ties(node_features)
        logger.info(
            f"Obtained node features of shape: {node_features.shape} and ties of shape: {ties.shape}"
        )
        return node_features, ties

    def update_ties(self, node_features):
        data = node_features.copy()
        data["source"] = self._focal_node
        data["target"] = data["title"]
        data["weight"] = 1
        return data[["source", "target", "weight"]].drop_duplicates()

    def update_tie_features(self):
        pass

    def update_node_features(self):
        data = pd.concat(
            [
                get_shelf_data(user_id=self._focal_node, shelf="read"),
                get_shelf_data(
                    user_id=self._focal_node,
                    shelf="currently-reading",
                    date_key="added",
                ),
            ]
        )
        return data
