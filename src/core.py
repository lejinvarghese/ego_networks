# -*- coding: utf-8 -*-
"""
Core abstract functionalities for the main objects in the multi layered complex network.
"""
from abc import ABC, abstractmethod, abstractproperty


class EgoNeighborhood(ABC):
    """
    Abstract class for ego neighborhood for any given layer
    """

    @abstractproperty
    def layer(self):
        pass

    @abstractproperty
    def focal_node(self):
        pass

    @abstractproperty
    def max_radius(self):
        pass

    @abstractmethod
    def update_neighborhood(self):
        pass

    @abstractmethod
    def update_ties(self):
        pass

    @abstractmethod
    def update_tie_features(self):
        pass

    @abstractmethod
    def update_node_features(self):
        pass


class EgoNetwork(ABC):
    """
    Abstract class for ego network spanning across all layers
    """

    @abstractproperty
    def n_layers(self):
        pass

    @abstractproperty
    def focal_node_id(self):
        pass

    @abstractproperty
    def radius(self):
        pass

    @abstractmethod
    def create_measures(self, network):
        pass


class NetworkMeasures(ABC):
    """
    Abstract class for ego network measures
    """

    @abstractproperty
    def summary_measures(self):
        pass

    @abstractproperty
    def node_measures(self):
        pass

    @abstractproperty
    def edge_measures(self):
        pass


class NetworkRecommender(ABC):
    """
    Abstract class for network recommenders
    """

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def test():
        pass

    @abstractmethod
    def get_recommendations(k):
        pass
