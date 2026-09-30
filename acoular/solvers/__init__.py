# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Abstract solver and problem interfaces for inverse methods.

.. autosummary::
    :toctree: generated/

    backends
    base
    problems
    scalings
    solver
    BaseProblem
    SolverOutput
    LeastSquaresProblem
    NNLSSolver
    OMPCVSolver
    L1RegularizedLeastSquaresProblem
    LassoLarsSolver
    LassoLarsBICSolver
    SolverBase
    LeastSquaresSolver
    LBFGSBSolver
    FISTALassoSolver
    SplitBregmanLassoSolver
    register_problem_scaling
    get_problem_scaling
    available_problem_scalings
    register_solver_backend
    get_solver_backend
    registered_solver_backends
    available_solver_backends
    solver_backend_info
    sklearn_backends
    scipy_backends
    pylops_backends
"""

from . import pylops_backends, scipy_backends, sklearn_backends  # noqa: F401
from .backends import (
    available_solver_backends,
    get_solver_backend,
    register_solver_backend,
    registered_solver_backends,
    solver_backend_info,
)
from .base import SolverBase, SolverOutput
from .problems import BaseProblem, L1RegularizedLeastSquaresProblem, LeastSquaresProblem
from .scalings import available_problem_scalings, get_problem_scaling, register_problem_scaling
from .solver import (
    FISTALassoSolver,
    LassoLarsBICSolver,
    LassoLarsSolver,
    LBFGSBSolver,
    LeastSquaresSolver,
    NNLSSolver,
    OMPCVSolver,
    SplitBregmanLassoSolver,
)

__all__ = [
    'BaseProblem',
    'FISTALassoSolver',
    'L1RegularizedLeastSquaresProblem',
    'LBFGSBSolver',
    'LassoLarsBICSolver',
    'LassoLarsSolver',
    'LeastSquaresProblem',
    'LeastSquaresSolver',
    'NNLSSolver',
    'OMPCVSolver',
    'SolverBase',
    'SolverOutput',
    'SplitBregmanLassoSolver',
    'available_problem_scalings',
    'available_solver_backends',
    'get_problem_scaling',
    'get_solver_backend',
    'register_problem_scaling',
    'register_solver_backend',
    'registered_solver_backends',
    'solver_backend_info',
]
