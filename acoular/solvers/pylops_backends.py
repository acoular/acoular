# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""PyLops-backed solver backend implementations."""

from .backends import register_solver_backend
from .base import SolverOutput


# ------------------------------------------------------------------------------
def pylops_fista_lasso_backend(solver, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG001
    """Solve L1-regularized least squares via PyLops FISTA.

    Reproduces the current CMF FISTA behavior. Unlike sklearn Lasso-style
    wrappers, `problem.alpha` is used unchanged, not multiplied by unit_mult.

    Parameters
    ----------
    solver : FISTALassoSolver
        Solver instance carrying :attr:`~FISTALassoSolver.backend_kwargs`.
    problem : L1RegularizedLeastSquaresProblem
        Problem instance carrying `alpha`.
    dictionary_matrix : array-like of shape (n_data, n_sources)
        Scaled dictionary matrix for one frequency bin.
    data : array-like of shape (n_data,)
        Scaled measurement data for one frequency bin.
    start_value : array-like of shape (n_sources,), optional
        Unused by this backend.

    Returns
    -------
    SolverOutput
        Solved source strengths in `.solution`; iteration/cost diagnostics in `.info`.
    """
    from pylops import MatrixMult
    from pylops.optimization.sparsity import fista

    operator = MatrixMult(dictionary_matrix)
    kwargs = {'tol': 1e-10, 'show': False}
    kwargs.update(solver.backend_kwargs)
    solution, iterations, cost = fista(Op=operator, y=data, eps=problem.alpha, alpha=None, **kwargs)
    return SolverOutput(solution=solution, info={'iterations': iterations, 'scaled_cost': cost})


register_solver_backend('fista_lasso', 'pylops', pylops_fista_lasso_backend, dependency='pylops')


# ------------------------------------------------------------------------------
def pylops_split_bregman_lasso_backend(solver, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG001
    """Solve L1-regularized least squares via PyLops Split Bregman.

    Reproduces the current CMF Split_Bregman behavior. Like FISTA, `problem.alpha`
    is used unchanged here, not multiplied by unit_mult.

    Parameters
    ----------
    solver : SplitBregmanLassoSolver
        Solver instance carrying :attr:`~SplitBregmanLassoSolver.backend_kwargs`.
    problem : L1RegularizedLeastSquaresProblem
        Problem instance carrying `alpha`.
    dictionary_matrix : array-like of shape (n_data, n_sources)
        Scaled dictionary matrix for one frequency bin.
    data : array-like of shape (n_data,)
        Scaled measurement data for one frequency bin.
    start_value : array-like of shape (n_sources,), optional
        Unused by this backend.

    Returns
    -------
    SolverOutput
        Solved source strengths in `.solution`; iteration/cost diagnostics in `.info`.
    """
    from pylops import Identity, MatrixMult
    from pylops.optimization.sparsity import splitbregman

    num_points = dictionary_matrix.shape[1]
    operator = MatrixMult(dictionary_matrix)
    regularizer = problem.alpha * Identity(num_points)
    kwargs = {
        'niter_inner': 5,
        'RegsL2': None,
        'dataregsL2': None,
        'mu': 1.0,
        'epsRL1s': [1],
        'tol': 1e-10,
        'tau': 1.0,
        'show': False,
    }
    kwargs.update(solver.backend_kwargs)
    solution, iterations, cost = splitbregman(Op=operator, RegsL1=[regularizer], y=data, **kwargs)
    return SolverOutput(solution=solution, info={'iterations': iterations, 'scaled_cost': cost})


register_solver_backend('split_bregman_lasso', 'pylops', pylops_split_bregman_lasso_backend, dependency='pylops')
