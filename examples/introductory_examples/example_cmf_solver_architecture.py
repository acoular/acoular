# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""CMF solver architecture -- Problem and Solver, composed.
============================================================

:class:`~acoular.fbeamform.BeamformerCMF` fits a modeled cross-spectral
matrix to measured data for each frequency bin. Solving that fit is split
into two parts: a :class:`~acoular.solvers.LeastSquaresProblem` describes
*what* is being solved (non-negativity, scaling, an optional L1 penalty),
while a :class:`~acoular.solvers.SolverBase` describes *how* the numbers
are actually computed. Choosing a built-in ``method=`` on ``BeamformerCMF``
just configures one of each.

This example first shows what that configuration can do, then shows that
every part of it is extensible: a new solver class, a new scaling, and a
new numerical backend can each be added without changing BeamformerCMF or
any of the existing solvers.
"""

from acoular.solvers import (
    LeastSquaresProblem,
    LeastSquaresSolver,
    NNLSSolver,
    SolverOutput,
)
from acoular.solvers.backends import register_solver_backend
from acoular.solvers.scalings import register_problem_scaling, unit_l2_scaling

import numpy as np
from sklearn.linear_model import LinearRegression

# %%
# A single-frequency-bin CMF problem: ``dictionary_matrix @ x = data``.
# ``A`` stands in for the steering-vector dictionary, ``data`` for one row
# of the vectorized cross-spectral matrix.
A = np.random.randn(40, 12)
x_true = np.zeros(12)
x_true[[2, 7, 11]] = [3.0, 1.5, 6]
data = A @ x_true + 0.01 * np.random.randn(40)
unit = 1e9


# %%
# Internal check, not part of the published user guide: confirming that
# ``Problem.solve()`` matches a plain scikit-learn fit done by hand (scale
# the dictionary columns, fit, rescale the result back out).
A_s, scale = unit_l2_scaling(A, axis=0)
x_old = LinearRegression(positive=True).fit(A_s, data * unit).coef_ / scale / unit
x_new = (
    LeastSquaresProblem(solver=NNLSSolver(), nonnegative=True, dictionary_scaling='unit_l2', unit_mult=unit)
    .solve(A, data)
    .solution
)
print('Equivalent to the old inline code:', np.allclose(x_old, x_new))

# %%
# A method's numerical library is a backend choice, not something baked
# into the method itself. The same ``'nnls'`` method has two registered
# backends here; choosing between them is one keyword.
same_via_sklearn = LeastSquaresProblem(
    solver=NNLSSolver(backend='sklearn', backend_kwargs={'fit_intercept': False}),
    nonnegative=True,
    dictionary_scaling='unit_l2',
    unit_mult=unit,
)
same_via_scipy = LeastSquaresProblem(
    solver=NNLSSolver(backend='scipy'), nonnegative=True, dictionary_scaling='unit_l2', unit_mult=unit
)
x_sklearn = same_via_sklearn.solve(A, data).solution
x_scipy = same_via_scipy.solve(A, data).solution
print('Same method, two backends, same answer:', np.allclose(x_sklearn, x_scipy))


# %%
# Adding a solver that was never one of the 7 built-in methods only needs a
# subclass of :class:`~acoular.solvers.LeastSquaresSolver` implementing
# ``solve()`` and returning a :class:`~acoular.solvers.SolverOutput`.
class RidgeSolver(LeastSquaresSolver):
    def solve(self, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG002
        n = dictionary_matrix.shape[1]
        sol = np.linalg.solve(dictionary_matrix.T @ dictionary_matrix + 0.1 * np.eye(n), dictionary_matrix.T @ data)
        return SolverOutput(solution=sol)


custom = LeastSquaresProblem(solver=RidgeSolver(), dictionary_scaling='unit_l2', unit_mult=unit)
print('Custom solver, same interface:', np.round(custom.solve(A, data).solution, 2))


# %%
# A custom scaling is just a function with the same
# ``(array, axis) -> (scaled_array, scale)`` signature as the built-ins,
# registered under a name.


@register_problem_scaling('max_abs')
def max_abs_scaling(array, axis=None):
    scale = np.max(np.abs(array), axis=axis)
    return array / scale, scale


custom_scaling = LeastSquaresProblem(
    solver=NNLSSolver(), nonnegative=True, dictionary_scaling='max_abs', unit_mult=unit
)
print('Custom scaling:', np.round(custom_scaling.solve(A, data).solution, 2))


# %%
# A custom backend is a function with the same
# ``(solver, problem, dictionary_matrix, data, start_value) -> SolverOutput``
# signature as the built-ins, registered for an existing solver family.
# (Illustrative only -- lstsq + clipping is not a correct NNLS solver.)


def naive_nnls_backend(solver, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG001
    solution, *_ = np.linalg.lstsq(dictionary_matrix, data, rcond=None)
    return SolverOutput(solution=np.clip(solution, 0, None))


register_solver_backend('nnls', 'naive', naive_nnls_backend)
custom_backend = LeastSquaresProblem(
    solver=NNLSSolver(backend='naive'), nonnegative=True, dictionary_scaling='unit_l2', unit_mult=unit
)
print('Custom backend:', np.round(custom_backend.solve(A, data).solution, 2))
