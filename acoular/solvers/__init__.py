# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Solver backends for inverse methods.

This subpackage provides abstract base classes and backend-specific
implementations that can be injected into inverse methods such as
:class:`acoular.fbeamform.BeamformerCMF`.

.. autosummary::
    :toctree: ../generated/

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

from .base import LeastSquaresSolver, SolverBase
from .pylops import ISTACV, PylopsLeastSquaresSolver

__all__ = [
    'SolverBase',
    'LeastSquaresSolver',
    'PylopsLeastSquaresSolver',
    'ISTACV',
]

if config.have_pylops:
    from .pylops import (
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
