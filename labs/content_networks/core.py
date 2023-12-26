"""
Core abstract functionalities.
"""
from abc import ABC, abstractproperty
from enum import Enum
from typing import List, Float, String


class Meme(ABC):
    @abstractproperty
    def summary(self) -> String:
        pass


class MemeticExpression(Enum):
    explicit = 0
    implicit = 1
    ironic = 2
    post_ironic = 3
    meta_ironic = 4


class Idea:
    def __init__(
        self,
        memes: List[Meme],
        expressions: List[MemeticExpression],
        strengths: List[Float],
    ):
        self.memes = memes
        self.expressions = expressions
        self.strengths = strengths


class Content(ABC):
    @abstractproperty
    def id(self) -> String:
        pass

    @abstractproperty
    def name(self) -> String:
        pass

    @abstractproperty
    def summary(self) -> String:
        pass

    @abstractproperty
    def format(self) -> String:
        pass

    @abstractproperty
    def ideas(self) -> List[Idea]:
        pass
