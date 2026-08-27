# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Abstract solver and problem interfaces for inverse methods.

.. autosummary::
    :toctree: generated/

    base
    problems
    scalings
    solver
    BaseProblem
    LeastSquaresProblem
    L1RegularizedLeastSquaresProblem
    SolverBase
    LeastSquaresSolver
    register_problem_scaling
    get_problem_scaling
    available_problem_scalings
"""

from .base import SolverBase
from .problems import BaseProblem, L1RegularizedLeastSquaresProblem, LeastSquaresProblem
from .scalings import available_problem_scalings, get_problem_scaling, register_problem_scaling
from .solver import LeastSquaresSolver

__all__ = [
    'BaseProblem',
    'L1RegularizedLeastSquaresProblem',
    'LeastSquaresProblem',
    'LeastSquaresSolver',
    'SolverBase',
    'available_problem_scalings',
    'get_problem_scaling',
    'register_problem_scaling',
]
