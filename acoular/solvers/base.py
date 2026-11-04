# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Abstract solver backend interfaces for inverse problems."""

from abc import abstractmethod

from acoular.internal import digest

from traits.api import ABCHasStrictTraits, Dict, Property, Str, cached_property


class SolverBase(ABCHasStrictTraits):
    """Common interface for backend-specific solver implementations."""

    backend = Str()
    extra_backend_kwargs = Dict()
    digest = Property(depends_on=['backend', 'extra_backend_kwargs'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @abstractmethod
    def solve(self, problem, dictionary_matrix, y, x=None, index=None):
        """Solve the inverse problem for one frequency bin."""
