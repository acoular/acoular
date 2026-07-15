# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for solver backend interfaces."""

import pytest


def test_only_solver_base_classes_are_publicly_available():
    """The solver package exposes only the backend concept, no concrete backends."""
    import acoular.solvers as solvers

    assert solvers.__all__ == ['SolverBase', 'LeastSquaresSolver']
    assert hasattr(solvers, 'SolverBase')
    assert hasattr(solvers, 'LeastSquaresSolver')
    assert not hasattr(solvers, 'PylopsLeastSquaresSolver')
    assert not hasattr(solvers, 'ISTACV')


def test_pylops_solver_backend_is_not_available():
    """PyLops must not be importable as an Acoular solver backend yet."""
    with pytest.raises(ModuleNotFoundError):
        __import__('acoular.solvers.pylops')
