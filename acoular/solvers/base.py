# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Base classes for inverse solver backends.

This module defines abstract solver interfaces that can be injected into inverse
methods such as :class:`acoular.fbeamform.BeamformerCMF`.

.. autosummary::
    :toctree: ../generated/

    SolverBase
    LeastSquaresSolver
"""

from abc import abstractmethod

import numpy as np
from traits.api import ABCHasStrictTraits, Any, Array, Dict, Float, Property, cached_property

from acoular.internal import digest


class SolverBase(ABCHasStrictTraits):
    """Common base class for all solver backends.

    Notes
    -----
    Solver instances are intended to be passed to inverse beamformers via their
    ``solver`` trait. Implementations must provide the :meth:`solve` method and
    may store additional run diagnostics in :attr:`output`.
    """

    #: Optional container for solver output and diagnostics.
    output = Any

    #: Keyword options forwarded to the underlying backend solver.
    options = Dict

    #: A unique identifier for the solver, based on its properties. (read-only)
    digest = Property(depends_on=['options'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @abstractmethod
    def solve(self, A, y, x, index):  # noqa: N803
        """Solve the linear inverse problem for one frequency bin.

        Parameters
        ----------
        A : array_like
            Sensing matrix of the inverse problem.
        y : array_like
            Measurement vector (or matrix) for the current frequency.
        x : array_like
            Initial solution estimate.
        index : int
            Frequency index currently being processed.

        Returns
        -------
        array_like
            Estimated source strengths for the current frequency.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        msg = 'The solve method must be implemented by subclasses of SolverBase.'
        raise NotImplementedError(msg)


class LeastSquaresSolver(SolverBase):
    """Base class for least-squares style solver implementations.

    This class exists as a semantic base type for inverse methods that accept
    custom least-squares-like solvers.
    """

    #: Normalization factors for sensing matrix columns. Applied internally
    #: by the solver to improve numerical conditioning. Typically computed
    #: as column-wise L2 norms of the sensing matrix.
    norms = Array(dtype=float, value=np.array([1.0]))

    #: Unit multiplier for measurement vector scaling. Applied internally by
    #: the solver to avoid numerical precision issues. The solver automatically
    #: inverts this scaling in the returned solution.
    unit = Float(1.0)

    #: A unique identifier for the solver, based on its properties. (read-only)
    digest = Property(depends_on=['options', 'norms', 'unit'])

    @cached_property
    def _get_digest(self):
        return digest(self)
