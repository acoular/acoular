from traits.api import ABCHasStrictTraits, Str, Dict, Property, cached_property
from abc import abstractmethod
from acoular.internal import digest
from .base import SolverBase

class LeastSquaresSolver(SolverBase):
    'Semantic base class for least-squares-style solvers'
    

