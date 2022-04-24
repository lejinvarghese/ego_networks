# -*- coding: utf-8 -*-
"""
Extracts Twitter API v2.0 to extract forward one hop neighbors of followed users
"""
from abc import ABC, abstractmethod, abstractproperty


class EgoNeighborhood(ABC):
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
    @abstractproperty
    def n_layers(self):
        pass

    @abstractproperty
    def focal_node(self):
        pass

    @abstractproperty
    def radius(self):
        pass

    @abstractmethod
    def calculate_metrics(self, network):
        pass


class EgoNetworkRecommender(EgoNetwork):
    pass
