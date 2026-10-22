# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""Scikit-learn-backed solver backend implementations.

.. autosummary::
    :toctree: generated/

    sklearn_nnls_backend
    sklearn_lasso_lars_backend
    sklearn_lasso_lars_bic_backend
    sklearn_ompcv_backend
"""

from .backends import register_solver_backend
from .base import SolverOutput

from sklearn.linear_model import LassoLars, LassoLarsIC, LinearRegression, OrthogonalMatchingPursuitCV


# ------------------------------------------------------------------------------
def sklearn_nnls_backend(solver, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG001
    """Solve non-negative least squares via scikit-learn's LinearRegression.

    Reproduces the current CMF NNLS behavior.

    Parameters
    ----------
    solver : NNLSSolver
        Solver instance carrying :attr:`~NNLSSolver.backend_kwargs`.
    problem : LeastSquaresProblem
        Problem instance carrying :attr:`~acoular.solvers.LeastSquaresProblem.nonnegative`.
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
    model = LinearRegression(positive=problem.nonnegative, **solver.backend_kwargs)
    model.fit(dictionary_matrix, data)
    return SolverOutput(solution=model.coef_, info={})


# default backend for NNLSSolver (matches _backend's default in solver.py) --
# required to reproduce legacy CMF NNLS behavior exactly, per equivalence tests
register_solver_backend('nnls', 'sklearn', sklearn_nnls_backend, dependency='sklearn')


# ------------------------------------------------------------------------------
def sklearn_lasso_lars_backend(solver, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG001
    """Solve L1-regularized least squares via scikit-learn's LassoLars.

    Reproduces the current CMF LassoLars behavior, including the legacy
    ``alpha * unit_mult`` convention.

    Parameters
    ----------
    solver : LassoLarsSolver
        Solver instance carrying :attr:`~LassoLarsSolver.backend_kwargs`.
    problem : L1RegularizedLeastSquaresProblem
        Problem instance carrying `alpha`, `nonnegative`, and `unit_mult`.
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
    model = LassoLars(
        alpha=problem.alpha * problem.unit_mult,
        positive=problem.nonnegative,
        **solver.backend_kwargs,
    )
    model.fit(dictionary_matrix, data)
    return SolverOutput(solution=model.coef_, info={})


# default (and currently only) backend for LassoLarsSolver --
# required to reproduce legacy CMF LassoLars behavior exactly
register_solver_backend('lasso_lars', 'sklearn', sklearn_lasso_lars_backend, dependency='sklearn')


# ------------------------------------------------------------------------------
def sklearn_lasso_lars_bic_backend(solver, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG001
    """Solve L1-regularized least squares via scikit-learn's BIC-selected LassoLarsIC.

    Reproduces the current CMF LassoLarsBIC behavior. `problem.alpha` is not used
    here: LassoLarsIC selects its own regularization strength internally via the
    BIC criterion, rather than taking a user-specified alpha.

    Parameters
    ----------
    solver : LassoLarsBICSolver
        Solver instance carrying :attr:`~LassoLarsBICSolver.backend_kwargs`.
    problem : L1RegularizedLeastSquaresProblem
        Problem instance carrying `nonnegative`. `alpha` is not used by this backend.
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
    model = LassoLarsIC(criterion='bic', positive=problem.nonnegative, **solver.backend_kwargs)
    model.fit(dictionary_matrix, data)
    return SolverOutput(solution=model.coef_, info={})


# default (and currently only) backend for LassoLarsBICSolver --
# required to reproduce legacy CMF LassoLarsBIC behavior exactly
register_solver_backend('lasso_lars_bic', 'sklearn', sklearn_lasso_lars_bic_backend, dependency='sklearn')


# ------------------------------------------------------------------------------
def sklearn_ompcv_backend(solver, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG001
    """Solve via scikit-learn's cross-validated Orthogonal Matching Pursuit.

    Reproduces the current CMF OMPCV behavior. Unlike the other sklearn-backed
    families, this algorithm has no non-negativity option, so `problem.nonnegative`
    is not used here.

    Parameters
    ----------
    solver : OMPCVSolver
        Solver instance carrying :attr:`~OMPCVSolver.backend_kwargs`.
    problem : LeastSquaresProblem
        Unused by this backend; OMPCV has no non-negativity constraint to apply.
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
    model = OrthogonalMatchingPursuitCV(**solver.backend_kwargs)
    model.fit(dictionary_matrix, data)
    return SolverOutput(solution=model.coef_, info={})


# default (and currently only) backend for OMPCVSolver --
# required to reproduce legacy CMF OMPCV behavior exactly
register_solver_backend('ompcv', 'sklearn', sklearn_ompcv_backend, dependency='sklearn')
