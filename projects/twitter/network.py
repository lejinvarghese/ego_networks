from abc import ABC, abstractmethod, abstractproperty
from distutils import core


class EgoNetwork(ABC):
    @abstractproperty
    def core_user(self):
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
    def __init__(self, core_user: str):
        self._core_user = core_user

    @property
    def core_user(self):
        return self._core_user

    def authenticate(self):
        pass

    def retrieve_edges(self, radius):
        pass

    def retrieve_nodes(self):
        pass

    def create_network(self):
        pass


if __name__ == "__main__":
    t = TwitterEgoNetwork(core_user=123)
    print(t.core_user)
