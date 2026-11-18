# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Solver backends for least-squares-style inverse problems.

.. autosummary::
    :toctree: generated/

    LeastSquaresSolver
    BackendDispatchSolver
    NNLSSolver
    LassoLarsSolver
    LassoLarsBICSolver
    OMPCVSolver
    LBFGSBSolver
    FISTALassoSolver
    SplitBregmanLassoSolver
"""

from abc import abstractmethod

from acoular.internal import digest

from .backends import get_solver_backend
from .base import SolverBase

from traits.api import Dict, Property, Str, cached_property


class LeastSquaresSolver(SolverBase):
    """Semantic base class for least-squares-style solvers."""


class BackendDispatchSolver(LeastSquaresSolver):
    """Base class for solvers that dispatch to a registered backend by family name."""

    @abstractmethod
    def _family(self):
        """Solver family used for backend registry lookups. Subclasses must override this."""

    #: Private storage for :attr:`backend`.
    # Subclasses override the default via re-declaration.
    _backend = Str()

    #: Name of the registered backend implementation to use.
    backend = Property(depends_on=['_backend'])

    #: Additional backend-specific keyword arguments.
    backend_kwargs = Dict()

    #: Unique identifier for this solver configuration. (read-only)
    digest = Property(depends_on=['_backend', 'backend_kwargs'])

    def _get_backend(self):
        return self._backend

    def _set_backend(self, value):
        get_solver_backend(self._family, value)
        self._backend = value

    @cached_property
    def _get_digest(self):
        return digest(self)

    def solve(self, problem, dictionary_matrix, data, start_value=None):
        """Solve the problem by dispatching to the selected registered backend.

        Parameters
        ----------
        problem : BaseProblem
         Problem instance that delegates to this solver.
        dictionary_matrix : array-like of shape (n_data, n_sources)
            Scaled dictionary matrix for one frequency bin.
        data : array-like of shape (n_data,)
            Scaled measurement data for one frequency bin.
        start_value : array-like of shape (n_sources,), optional
            Solver-specific numerical start value.

        Returns
        -------
        SolverOutput
            Solved source strengths (in `.solution`) plus optional backend
            diagnostics (in `.info`), in the scaled problem coordinates.
        """
        backend_func = get_solver_backend(self._family, self.backend)
        return backend_func(self, problem, dictionary_matrix, data, start_value)


class NNLSSolver(BackendDispatchSolver):
    """Non-negative least-squares solver.

    Dispatches to a registered backend.
    """

    _family = 'nnls'
    _backend = Str('sklearn')


class LassoLarsSolver(BackendDispatchSolver):
    """Lasso (L1-regularized least squares) solver via LARS.

    Dispatches to a registered backend.
    """

    _family = 'lasso_lars'
    _backend = Str('sklearn')


class LassoLarsBICSolver(BackendDispatchSolver):
    """Lasso solver via LARS with BIC-selected regularization strength.

    Dispatches to a registered backend.
    """

    _family = 'lasso_lars_bic'
    _backend = Str('sklearn')


class OMPCVSolver(BackendDispatchSolver):
    """Orthogonal Matching Pursuit (cross-validated) solver.

    Dispatches to a registered backend.
    """

    _family = 'ompcv'
    _backend = Str('sklearn')


class LBFGSBSolver(BackendDispatchSolver):
    """L-BFGS-B constrained optimization solver.

    Dispatches to a registered backend.
    """

    _family = 'lbfgsb'
    _backend = Str('scipy')


class FISTALassoSolver(BackendDispatchSolver):
    """L1-regularized least squares solver via PyLops FISTA.

    Dispatches to a registered backend.
    """

    _family = 'fista_lasso'
    _backend = Str('pylops')


class SplitBregmanLassoSolver(BackendDispatchSolver):
    """L1-regularized least squares solver via PyLops Split Bregman.

    Dispatches to a registered backend.
    """

    _family = 'split_bregman_lasso'
    _backend = Str('pylops')
