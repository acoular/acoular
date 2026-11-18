# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""SciPy-backed solver backend implementations."""

from .backends import register_solver_backend
from .base import SolverOutput

import numpy as np
from scipy.optimize import fmin_l_bfgs_b, nnls


# ------------------------------------------------------------------------------
def scipy_nnls_backend(solver, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG001
    """Solve non-negative least squares via :func:`scipy.optimize.nnls`.

    Parameters
    ----------
    solver : NNLSSolver
        Solver instance carrying :attr:`~NNLSSolver.backend_kwargs`.
    problem : BaseProblem
        Unused by this backend; :func:`scipy.optimize.nnls` is always non-negative.
    dictionary_matrix : array-like of shape (n_data, n_sources)
        Scaled dictionary matrix for one frequency bin.
    data : array-like of shape (n_data,)
        Scaled measurement data for one frequency bin.
    start_value : array-like of shape (n_sources,), optional
        Unused by this backend.

    Returns
    -------
    SolverOutput
        Solved source strengths in `.solution`.
    """
    solution, residual = nnls(dictionary_matrix, data, **solver.backend_kwargs)
    return SolverOutput(solution=solution, info={})


# second registered backend for 'nnls', alongside sklearn_backends.py's 'sklearn' one
register_solver_backend('nnls', 'scipy', scipy_nnls_backend, dependency='scipy')


# ------------------------------------------------------------------------------
def scipy_lbfgsb_backend(solver, problem, dictionary_matrix, data, start_value=None):
    """Solve via scipy's L-BFGS-B constrained optimizer.

    Reproduces the current CMF fmin_l_bfgs_b behavior.

    Parameters
    ----------
    solver : LBFGSBSolver
        Solver instance carrying :attr:`~LBFGSBSolver.backend_kwargs`.
    problem : LeastSquaresProblem
        Problem instance carrying `nonnegative`.
    dictionary_matrix : array-like of shape (n_data, n_sources)
        Scaled dictionary matrix for one frequency bin.
    data : array-like of shape (n_data,)
        Scaled measurement data for one frequency bin.
    start_value : array-like of shape (n_sources,), optional
        Initial guess. Defaults to an all-ones vector if not given.

    Returns
    -------
    SolverOutput
        Solved source strengths in `.solution`; iteration count, convergence
        status, and objective value diagnostics in `.info`.
    """
    num_points = dictionary_matrix.shape[1]
    data_column = data[:, np.newaxis]

    def function(x):
        func = x.T @ dictionary_matrix.T @ dictionary_matrix @ x - 2 * data.T @ dictionary_matrix @ x + data.T @ data
        der = 2 * dictionary_matrix.T @ dictionary_matrix @ x.T[:, np.newaxis] - 2 * dictionary_matrix.T @ data_column
        return func, der[:, 0]

    x0 = start_value if start_value is not None else np.ones(num_points)
    lower = 0.0 if problem.nonnegative else -np.inf
    bounds = np.tile((lower, np.inf), (num_points, 1))
    kwargs = {
        'fprime': None,
        'approx_grad': 0,
        'bounds': bounds,
        'm': 10,
        'factr': 10000000.0,
        'pgtol': 1e-05,
        'epsilon': 1e-08,
        'maxfun': 15000,
        'maxls': 20,
    }
    kwargs.update(solver.backend_kwargs)
    solution, yval, info = fmin_l_bfgs_b(function, x0, **kwargs)
    return SolverOutput(solution=solution, info={'nit': info['nit'], 'warnflag': info['warnflag'], 'scaled_yval': yval})


register_solver_backend('lbfgsb', 'scipy', scipy_lbfgsb_backend, dependency='scipy')
