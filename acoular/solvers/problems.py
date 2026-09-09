# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Abstract problem interfaces for inverse methods."""

from acoular.internal import digest

from .solver import SolverBase

import scipy.linalg as spla
from traits.api import ABCHasStrictTraits, Bool, Enum, Instance, Property, Range, cached_property


class BaseProblem(ABCHasStrictTraits):
    """Common interface for inverse problems that delegate solving to an attached solver."""

    solver = Instance(SolverBase)
    digest = Property(depends_on=['solver.digest'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def solve(self, dictionary_matrix, y, x=None, index=None):
        """Solve for one frequency bin by delegating to the attached solver."""
        if self.solver is None:
            msg = 'No solver attached to this problem instance.'
            raise ValueError(msg)
        return self.solver.solve(self, dictionary_matrix, y, x=x, index=index)


class LeastSquaresProblem(BaseProblem):
    """Least-squares inverse problem with configurable dictionary/data normalization."""

    positive = Bool(False)
    dictionary_normalization = Enum('none', 'unit_l2')
    data_normalization = Enum('none')
    digest = Property(depends_on=['solver.digest', 'positive', 'dictionary_normalization', 'data_normalization'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def normalize_dictionary(self, dictionary_matrix):
        """Normalize sensing-matrix columns according to dictionary_normalization."""
        if self.dictionary_normalization == 'none':
            return dictionary_matrix, 1.0
        # Temporarily sets for 2 options in dictionary_normalization
        dict_norm = spla.norm(dictionary_matrix, axis=0)
        return dictionary_matrix / dict_norm, dict_norm

    def normalize_data(self, y):
        """Normalize the measurement vector according to data_normalization."""
        if self.data_normalization == 'none':
            return y, 1.0
        # Temporarily added to prevent an error until the function is implemented.
        msg = f'Unknown data_normalization: {self.data_normalization!r}'
        raise ValueError(msg)

    def solve(self, dictionary_matrix, y, x=None, index=None):
        """Normalize dictionary/data, solve via the attached solver, then rescale the result."""
        a_normal, dict_norm = self.normalize_dictionary(dictionary_matrix)
        y_normal, data_norm = self.normalize_data(y)
        result = super().solve(a_normal, y_normal, x, index)
        return result / dict_norm / data_norm


class L1RegularizedLeastSquaresProblem(LeastSquaresProblem):
    """Least-squares inverse problem with an added L1 regularization term."""

    alpha = Range(low=0.0, high=1.0, value=0)
    digest = Property(
        depends_on=['solver.digest', 'positive', 'dictionary_normalization', 'data_normalization', 'alpha']
    )

    @cached_property
    def _get_digest(self):
        return digest(self)
