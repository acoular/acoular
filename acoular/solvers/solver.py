# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Solver backends for least-squares-style inverse problems."""

from .base import SolverBase


class LeastSquaresSolver(SolverBase):
    """Semantic base class for least-squares-style solvers."""
