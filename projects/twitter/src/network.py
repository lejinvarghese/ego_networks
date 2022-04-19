from abc import ABC, abstractmethod, abstractproperty
from distutils import core


class EgoNetwork(ABC):
    @abstractproperty
    def focal_node(self):
        pass

    @abstractproperty
    def radius(self):
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
    def __init__(self, focal_node: int, radius: int):
        self._focal_node = int(focal_node)
        self._radius = int(radius)

    @property
    def focal_node(self):
        return self._focal_node

    @property
    def radius(self):
        return self._radius

    def authenticate(self):
        pass

    def retrieve_edges(self, radius):
        pass

    def retrieve_nodes(self):
        pass

    def create_network(self):
        pass


if __name__ == "__main__":
    pass
    # t_n = TwitterEgoNetwork(focal_node=123, radius=1)
    # print(t_n.focal_node)
    # print(t_n.radius)
