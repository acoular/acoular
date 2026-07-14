# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Compatibility module for solver classes.

The solver implementation has been moved to :mod:`acoular.solvers`.
This module re-exports the public classes for backward compatibility.

.. autosummary::
    :toctree: generated/

    SolverBase
    LeastSquaresSolver
    PylopsLeastSquaresSolver
    ISTACV
    CallbackBase
    SolutionResidualCallback
    SolutionResidualL1Callback
    RelativeResidualCallback
    IntermediateResultCallback
    AbsoluteResidualCallback
"""

from acoular.configuration import config

from .solvers import ISTACV, LeastSquaresSolver, PylopsLeastSquaresSolver, SolverBase

__all__ = [
    'SolverBase',
    'LeastSquaresSolver',
    'PylopsLeastSquaresSolver',
    'ISTACV',
]

if config.have_pylops:
    from .solvers import (
        AbsoluteResidualCallback,
        CallbackBase,
        IntermediateResultCallback,
        RelativeResidualCallback,
        SolutionResidualCallback,
        SolutionResidualL1Callback,
    )

    __all__ += [
        'CallbackBase',
        'IntermediateResultCallback',
        'SolutionResidualCallback',
        'SolutionResidualL1Callback',
        'RelativeResidualCallback',
        'AbsoluteResidualCallback',
    ]
