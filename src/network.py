# -*- coding: utf-8 -*-
"""
Extracts Twitter API v2.0 to extract forward one hop neighbors of followed users
"""
from abc import ABC, abstractproperty, abstractmethod


class EgoNetwork(ABC):
    @abstractproperty
    def focal_node(self):
        pass

    @abstractproperty
    def max_radius(self):
        pass

    @abstractmethod
    def retrieve_edges(self):
        pass

    @abstractmethod
    def retrieve_nodes(self):
        pass

    @abstractmethod
    def create_network(self):
        pass
