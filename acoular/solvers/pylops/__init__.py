# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""PyLops-backed solver implementations for :mod:`acoular.solvers`.

.. autosummary::
    :toctree: ../../generated/

    PylopsLeastSquaresSolver
    ISTACV
"""

from acoular.configuration import config

from .cv import ISTACV
from .solver import PylopsLeastSquaresSolver

__all__ = [
    'PylopsLeastSquaresSolver',
    'ISTACV',
]

if config.have_pylops:
    from .solver import (
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
