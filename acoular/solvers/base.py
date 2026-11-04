# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Abstract solver backend interfaces for inverse problems.

.. autosummary::
    :toctree: generated/

    SolverOutput
    SolverBase
"""

from abc import abstractmethod

from acoular.internal import digest

from traits.api import ABCHasStrictTraits, Any, Dict, Property, Str, cached_property


class SolverOutput(ABCHasStrictTraits):
    """Container for a solver's output.

    Keeps the solved source strengths together with optional backend-specific
    diagnostics. Deliberately does not carry a frequency index — mapping
    results to frequency bins is the responsibility of the calling beamformer.
    """

    #: Solved source strengths.
    solution = Any()

    #: Optional backend-specific diagnostics (e.g. status, iteration count,
    #: residual norms). Intentionally has no fixed schema.
    info = Dict()


class SolverBase(ABCHasStrictTraits):
    """Common interface for backend-specific solver implementations.

    Solver implementations receive an already assembled and scaled inverse
    problem from a :class:`~acoular.solvers.BaseProblem` instance and return the
    corresponding source strengths. They do not perform problem scaling.
    """

    #: Name of the numerical backend used by the solver.
    backend = Str()

    #: Additional backend-specific keyword arguments.
    backend_kwargs = Dict()

    #: Unique identifier for this solver configuration. (read-only)
    digest = Property(depends_on=['backend', 'backend_kwargs'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @abstractmethod
    def solve(self, problem, dictionary_matrix, data, start_value=None):
        """Solve the inverse problem for one frequency bin.

        The optional *start_value* is passed to solver implementations that can
        use a numerical start value. Its interpretation is solver-specific.

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
