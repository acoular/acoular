# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Solver interfaces for inverse methods.

This subpackage provides abstract base classes for solver implementations that
can be injected into inverse methods such as
:class:`acoular.fbeamform.BeamformerCMF`.

.. autosummary::
    :toctree: ../generated/

    SolverBase
    LeastSquaresSolver
"""

from .base import LeastSquaresSolver, SolverBase

__all__ = [
    'SolverBase',
    'LeastSquaresSolver',
]
