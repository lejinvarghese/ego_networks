# -*- coding: utf-8 -*-
"""
Extracts Twitter API v2.0 to extract forward one hop neighbors of followed users
"""
from abc import ABC, abstractmethod, abstractproperty


class EgoNetwork(ABC):
    @abstractproperty
    def focal_node(self):
        pass

    @abstractproperty
    def max_radius(self):
        pass

    @abstractmethod
    def create_network(self):
        pass

    @abstractmethod
    def update_neighborhood(self):
        pass

    @abstractmethod
    def retrieve_ties(self):
        pass

    @abstractmethod
    def retrieve_tie_features(self):
        pass

    @abstractmethod
    def retrieve_node_features(self):
        pass
