from traits.api import ABCHasStrictTraits, Str, Dict, Property, cached_property
from abc import abstractmethod
from acoular.internal import digest

class SolverBase(ABCHasStrictTraits):
    backend = Str()
    extra_backend_kwargs = Dict()
    digest = Property(depends_on = ['backend', 'extra_backend_kwargs'])

    @cached_property
    def _get_digest(self):
        return digest(self)
    
    @abstractmethod
    def solve(self, problem, A, y, x=None, index=None):
        'Solve the inverse problem for one frequency bin.'

