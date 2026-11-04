# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for the problem/solver interface abstractions and scaling registry."""

import acoular as ac
from acoular.solvers import (
    BaseProblem,
    L1RegularizedLeastSquaresProblem,
    LeastSquaresProblem,
    LeastSquaresSolver,
    NNLSSolver,
    SolverOutput,
    available_problem_scalings,
    available_solver_backends,
    get_problem_scaling,
    get_solver_backend,
    register_problem_scaling,
    register_solver_backend,
    registered_solver_backends,
    solver_backend_info,
)

import numpy as np
import pytest
from traits.api import Any


class _EchoSolver(LeastSquaresSolver):
    """Test double: returns the (scaled) data unchanged as the solution, and records call args.

    Only meaningful for testing data-side behavior (unit_mult, data_scaling, delegation) since
    it never actually inverts dictionary_matrix, so it cannot validate dictionary_scaling.
    """

    last_call = Any()

    def solve(self, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG002
        self.last_call = (problem, dictionary_matrix, data, start_value)
        return SolverOutput(solution=data, info={'echoed': True})


class _LstsqSolver(LeastSquaresSolver):
    """Solves via ordinary least squares; used to test dictionary-scaling round-trips."""

    def solve(self, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG002
        solution, *_ = np.linalg.lstsq(dictionary_matrix, data, rcond=None)
        return SolverOutput(solution=solution, info={})


# ---------------------------------------------------------------------------
# SolverOutput
# ---------------------------------------------------------------------------


def test_solver_output_holds_solution_and_info():
    output = SolverOutput(solution=np.array([1.0, 2.0]), info={'iterations': 3})
    np.testing.assert_allclose(output.solution, [1.0, 2.0])
    assert output.info == {'iterations': 3}


# ---------------------------------------------------------------------------
# SolverBase / digest
# ---------------------------------------------------------------------------


def test_solver_digest_changes_with_backend():
    solver = _EchoSolver()
    digest1 = solver.digest
    solver.backend = 'some_backend'
    assert solver.digest != digest1


def test_solver_digest_changes_with_backend_kwargs():
    solver = _EchoSolver()
    digest1 = solver.digest
    solver.backend_kwargs = {'max_iter': 10}
    assert solver.digest != digest1


# ---------------------------------------------------------------------------
# BaseProblem
# ---------------------------------------------------------------------------


def test_base_problem_raises_without_solver():
    problem = BaseProblem()
    with pytest.raises(ValueError, match='No solver attached'):
        problem.solve(np.zeros((2, 2)), np.zeros(2))


def test_base_problem_delegates():
    dictionary_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    data = np.array([5.0, 6.0])
    solver = _EchoSolver()
    problem = BaseProblem(solver=solver)

    result = problem.solve(dictionary_matrix, data)

    assert solver.last_call[0] is problem
    assert solver.last_call[1] is dictionary_matrix
    assert solver.last_call[2] is data
    assert solver.last_call[3] is None
    assert result.solution is data
    assert result.info == {'echoed': True}


def test_base_problem_forwards_start_value():
    dictionary_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    data = np.array([5.0, 6.0])
    x0 = np.array([0.1, 0.2])
    solver = _EchoSolver()
    problem = BaseProblem(solver=solver)

    problem.solve(dictionary_matrix, data, start_value=x0)

    assert solver.last_call[3] is x0


# ---------------------------------------------------------------------------
# LeastSquaresProblem
# ---------------------------------------------------------------------------


def test_least_squares_problem_solve_recovers_data_with_none_scaling():
    dictionary_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    data = np.array([5.0, 6.0])
    solver = _EchoSolver()
    problem = LeastSquaresProblem(solver=solver)  # defaults: 'none'/'none', unit_mult=1e9

    result = problem.solve(dictionary_matrix, data)

    np.testing.assert_allclose(result.solution, data)
    assert result.info == {'echoed': True}


def test_least_squares_problem_unit_l2_scaling_matches_unscaled_lstsq():
    dictionary_matrix = np.array([[1.0, 4.0], [3.0, 8.0], [2.0, 1.0]])  # non-uniform column norms
    data = np.array([5.0, 6.0, 1.0])
    solver = _LstsqSolver()
    problem = LeastSquaresProblem(solver=solver, dictionary_scaling='unit_l2', unit_mult=1.0)

    result = problem.solve(dictionary_matrix, data)

    expected, *_ = np.linalg.lstsq(dictionary_matrix, data, rcond=None)
    np.testing.assert_allclose(result.solution, expected)


def test_least_squares_problem_validates_scaling_name_immediately():
    problem = LeastSquaresProblem()
    with pytest.raises(ValueError):
        problem.dictionary_scaling = 'not_a_real_scaling'


def test_least_squares_problem_digest_changes_with_unit_mult():
    problem = LeastSquaresProblem()
    digest1 = problem.digest
    problem.unit_mult = 2e9
    assert problem.digest != digest1


def test_least_squares_problem_digest_changes_with_dictionary_scaling():
    problem = LeastSquaresProblem()
    digest1 = problem.digest
    problem.dictionary_scaling = 'unit_l2'
    assert problem.digest != digest1


def test_least_squares_problem_rejects_bad_shapes():
    problem = LeastSquaresProblem(solver=_EchoSolver())
    with pytest.raises(ValueError):
        problem.solve(np.zeros(3), np.zeros(3))  # dictionary_matrix not 2D
    with pytest.raises(ValueError):
        problem.solve(np.zeros((3, 2)), np.zeros((3, 1)))  # data not 1D
    with pytest.raises(ValueError):
        problem.solve(np.zeros((3, 2)), np.zeros(4))  # mismatched rows


# ---------------------------------------------------------------------------
# L1RegularizedLeastSquaresProblem
# ---------------------------------------------------------------------------


def test_l1_problem_digest_changes_with_alpha():
    problem = L1RegularizedLeastSquaresProblem()
    digest1 = problem.digest
    problem.alpha = 0.5
    assert problem.digest != digest1


# ---------------------------------------------------------------------------
# Scaling registry
# ---------------------------------------------------------------------------


def test_register_problem_scaling_decorator():
    @register_problem_scaling('test_decorator_scaling')
    def _scaling(array, axis=None):  # noqa: ARG001
        return array, 1.0

    assert 'test_decorator_scaling' in available_problem_scalings()
    assert get_problem_scaling('test_decorator_scaling') is _scaling


def test_register_problem_scaling_rejects_duplicate_name():
    def _scaling(array, axis=None):  # noqa: ARG001
        return array, 1.0

    register_problem_scaling('test_duplicate_scaling', _scaling)
    with pytest.raises(ValueError):
        register_problem_scaling('test_duplicate_scaling', _scaling)


def test_get_problem_scaling_unknown_name_raises():
    with pytest.raises(ValueError, match='not_registered_anywhere'):
        get_problem_scaling('not_registered_anywhere')


# ---------------------------------------------------------------------------
# Solver backend registry
# ---------------------------------------------------------------------------


def _dummy_backend(solver, problem, dictionary_matrix, data, start_value=None):  # noqa: ARG001
    return 'dummy result'


def test_register_solver_backend_and_lookup():
    register_solver_backend('test_family', 'test_backend', _dummy_backend, dependency=None)

    assert 'test_backend' in registered_solver_backends('test_family')
    assert get_solver_backend('test_family', 'test_backend') is _dummy_backend


def test_register_solver_backend_rejects_duplicate():
    register_solver_backend('test_family_dup', 'test_backend', _dummy_backend)
    with pytest.raises(ValueError):
        register_solver_backend('test_family_dup', 'test_backend', _dummy_backend)


def test_register_solver_backend_rejects_non_callable():
    with pytest.raises(TypeError):
        register_solver_backend('test_family_bad', 'test_backend', 'not_a_function')


def test_get_solver_backend_unknown_raises():
    with pytest.raises(ValueError, match='unknown_family'):
        get_solver_backend('unknown_family', 'unknown_backend')


def test_registered_vs_available_solver_backends():
    register_solver_backend(
        'test_family_avail', 'fake_backend', _dummy_backend, dependency='not_a_real_installed_package'
    )
    register_solver_backend('test_family_avail', 'real_backend', _dummy_backend, dependency=None)

    assert set(registered_solver_backends('test_family_avail')) == {'fake_backend', 'real_backend'}
    # fake_backend's dependency isn't importable, so it's registered but not available
    assert 'fake_backend' not in available_solver_backends('test_family_avail')
    assert 'real_backend' in available_solver_backends('test_family_avail')


def test_solver_backend_info_reports_dependency_and_availability():
    register_solver_backend('test_family_info', 'sklearn_like', _dummy_backend, dependency='sklearn')

    info = solver_backend_info('test_family_info', 'sklearn_like')

    assert info['dependency'] == 'sklearn'
    assert info['available'] is True


# ---------------------------------------------------------------------------
# NNLSSolver
# ---------------------------------------------------------------------------


def _small_cmf(method, **kwargs):
    mics = ac.MicGeom(pos_total=np.zeros((3, 2)))
    grid = ac.RectGrid(x_min=0, x_max=1, y_min=0, y_max=0, z=0.5, increment=1)
    steer = ac.SteeringVector(grid=grid, mics=mics)
    csm = np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]])
    freq_data = ac.PowerSpectraImport(csm=csm[np.newaxis, :, :], frequencies=np.array([1000.0]))
    bf = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=False, method=method, cached=False, **kwargs)
    return bf, csm


def test_nnls_solver_matches_legacy_cmf():
    bf, csm = _small_cmf('NNLS')
    legacy_q = bf.result[0]

    dictionary_matrix = bf._build_dictionary(1000.0)
    data = bf._vectorize_csm(csm).ravel()

    problem = LeastSquaresProblem(solver=NNLSSolver(), nonnegative=True)
    result = problem.solve(dictionary_matrix, data)

    np.testing.assert_allclose(result.solution, legacy_q)


def test_nnls_solver_scipy_backend_raises_on_bad_kwarg():
    dictionary_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    data = np.array([2.0, 3.0, 5.0])
    # atol isn't accepted by this scipy version's nnls() (only maxiter is) —
    # a real example of backend_kwargs drifting across scipy versions
    solver = NNLSSolver(backend='scipy', backend_kwargs={'atol': 1e-12})
    problem = LeastSquaresProblem(solver=solver, nonnegative=True)
    with pytest.raises(TypeError):
        problem.solve(dictionary_matrix, data)


def test_nnls_solver_sklearn_backend_raises_on_bad_kwarg():
    dictionary_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    data = np.array([2.0, 3.0, 5.0])
    # normalize was a real LinearRegression parameter, removed in sklearn >= 1.2 —
    # the legacy _calc code even has a comment about this exact removal
    solver = NNLSSolver(backend='sklearn', backend_kwargs={'normalize': True})
    problem = LeastSquaresProblem(solver=solver, nonnegative=True)
    with pytest.raises(TypeError):
        problem.solve(dictionary_matrix, data)
