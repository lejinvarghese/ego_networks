from abc import ABC, abstractmethod, abstractproperty
from distutils import core


class EgoNetwork(ABC):
    @abstractproperty
    def focal_node(self):
        pass

    @abstractmethod
    def authenticate(self):
        pass

    @abstractmethod
    def retrieve_edges(self, radius):
        pass

    @abstractmethod
    def retrieve_nodes(self):
        pass

    @abstractmethod
    def create_network(self):
        pass


class TwitterEgoNetwork(EgoNetwork):
    def __init__(self, focal_node: str):
        self._focal_node = focal_node

    @property
    def focal_node(self):
        return self._focal_node

    def authenticate(self):
        pass

    def retrieve_edges(self, radius):
        pass

    def retrieve_nodes(self):
        pass

    def create_network(self):
        pass


if __name__ == "__main__":
    t = TwitterEgoNetwork(focal_node=123)
    print(t.focal_node)
