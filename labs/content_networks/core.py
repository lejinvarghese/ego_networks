"""
Core abstract functionalities.
"""
from abc import ABC, property
from enum import Enum
from typing import List, Float, String

class Meme(ABC):
    @property
    def summary(self) -> String:
        pass

class MemeticExpression(Enum):
    explicit = 0
    implicit = 1
    ironic = 2
    post_ironic = 3
    meta_ironic = 4

class Idea:
    def __init__(self, memes: List[Meme], expressions:List[MemeticExpression], strengths: List[Float]):
        self.memes = memes
        self.expressions = expressions
        self.strengths = strengths

class Content(ABC):
    
    @property
    def id(self) -> String:
        pass
    
    @property
    def name(self) -> String:
        pass

    @property
    def summary(self) -> String:
        pass
    
    @property
    def format(self) -> String:
        pass
    
    @property
    def ideas(self) -> List[Idea]:
        pass
    

    
    
    