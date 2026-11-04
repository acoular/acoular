# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Abstract problem interfaces for inverse methods.

.. autosummary::
    :toctree: generated/

    BaseProblem
    LeastSquaresProblem
    L1RegularizedLeastSquaresProblem
"""

from acoular.internal import digest

from .base import SolverBase
from .scalings import get_problem_scaling

from traits.api import ABCHasStrictTraits, Bool, Float, Instance, Property, Range, Str, cached_property


class BaseProblem(ABCHasStrictTraits):
    """Common interface for inverse problems that delegate solving to an attached solver.

    A problem defines the mathematical interpretation of the assembled inputs.
    The attached solver performs the numerical solve.
    """

    #: Solver instance used to solve the assembled inverse problem.
    solver = Instance(SolverBase)

    #: Unique identifier for this problem configuration. (read-only)
    digest = Property(depends_on=['solver.digest'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def solve(self, dictionary_matrix, data, start_value=None):
        """Solve for one frequency bin by delegating to the attached solver.

        Parameters
        ----------
        dictionary_matrix : array-like
            Assembled dictionary matrix for one frequency bin.
        data : array-like
            Measurement data for one frequency bin.
        start_value : array-like, optional
            Solver-specific numerical start value.

        Returns
        -------
        SolverOutput
            Solver output produced by the attached solver.

        Raises
        ------
        ValueError
            If no solver is attached to this problem.
        """
        if self.solver is None:
            msg = 'No solver attached to this problem instance.'
            raise ValueError(msg)
        return self.solver.solve(self, dictionary_matrix, data, start_value=start_value)


class LeastSquaresProblem(BaseProblem):
    """Least-squares inverse problem with configurable dictionary/data scaling.

    The problem solves a single-frequency system ``dictionary_matrix * x = data``.
    Dictionary and data scaling are selected by names registered via
    :func:`~acoular.solvers.register_problem_scaling`.
    """

    #: If True, constrain the source strengths to be non-negative.
    nonnegative = Bool(False)

    #: Working-unit multiplier applied to scaled data before solving, preserving
    #: historical Acoular behavior. Not a registered scaling transform.
    unit_mult = Float(1e9)

    #: Private storage for :attr:`dictionary_scaling`.
    _dictionary_scaling = Str('none')

    #: Name of the scaling transform applied to dictionary columns.
    dictionary_scaling = Property(depends_on=['_dictionary_scaling'])

    #: Private storage for :attr:`data_scaling`.
    _data_scaling = Str('none')

    #: Name of the scaling transform applied to measurement data.
    data_scaling = Property(depends_on=['_data_scaling'])

    #: Unique identifier for this problem configuration. (read-only)
    digest = Property(depends_on=['solver.digest', 'nonnegative', '_dictionary_scaling', '_data_scaling', 'unit_mult'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _get_dictionary_scaling(self):
        get_problem_scaling(self._dictionary_scaling)
        return self._dictionary_scaling

    def _set_dictionary_scaling(self, value):
        get_problem_scaling(value)
        self._dictionary_scaling = value

    def _get_data_scaling(self):
        get_problem_scaling(self._data_scaling)
        return self._data_scaling

    def _set_data_scaling(self, value):
        get_problem_scaling(value)
        self._data_scaling = value

    def _validate_inputs(self, dictionary_matrix, data):
        """Validate the single-frequency least-squares input shapes."""
        if dictionary_matrix.ndim != 2:
            msg = 'dictionary_matrix must be 2-dimensional.'
            raise ValueError(msg)
        if data.ndim != 1:
            msg = 'data must be 1-dimensional.'
            raise ValueError(msg)
        if dictionary_matrix.shape[0] != data.shape[0]:
            msg = 'dictionary_matrix and data must have the same number of rows.'
            raise ValueError(msg)

    def scale_dictionary(self, dictionary_matrix):
        """Scale sensing-matrix columns according to :attr:`dictionary_scaling`.

        Parameters
        ----------
        dictionary_matrix : array-like of shape (n_data, n_sources)
            Dictionary matrix for one frequency bin.

        Returns
        -------
        array-like of shape (n_data, n_sources)
            Scaled dictionary matrix.
        float or array-like
            Dictionary scale factor used to recover source strengths.
        """
        scaling = get_problem_scaling(self.dictionary_scaling)
        return scaling(dictionary_matrix, axis=0)

    def scale_data(self, data):
        """Scale the measurement vector according to :attr:`data_scaling`.

        Parameters
        ----------
        data : array-like of shape (n_data,)
            Measurement data for one frequency bin.

        Returns
        -------
        array-like of shape (n_data,)
            Scaled measurement data.
        float or array-like
            Data scale factor used to recover source strengths.
        """
        scaling = get_problem_scaling(self.data_scaling)
        return scaling(data, axis=None)

    def solve(self, dictionary_matrix, data, start_value=None):
        """Scale dictionary/data, solve via the attached solver, then rescale the result.

        Also applies the working-unit multiplier (:attr:`unit_mult`) to the scaled
        data before solving, and divides it back out of the result afterward.
        The optional *start_value* is forwarded to the attached solver. It does
        not define a warm-start policy in the problem itself.

        Parameters
        ----------
        dictionary_matrix : array-like of shape (n_data, n_sources)
            Dictionary matrix for one frequency bin.
        data : array-like of shape (n_data,)
            Measurement data for one frequency bin.
        start_value : array-like of shape (n_sources,), optional
            Solver-specific numerical start value.

        Returns
        -------
        SolverOutput
            Solver output with `.solution` rescaled to the unscaled problem coordinates.

        Raises
        ------
        ValueError
            If the input shapes do not match the single-frequency least-squares
            contract, or if no solver is attached to this problem.
        """
        self._validate_inputs(dictionary_matrix, data)
        dictionary_matrix_scaled, dictionary_scale = self.scale_dictionary(dictionary_matrix)
        data_scaled, data_scale = self.scale_data(data)

        data_work = data_scaled * self.unit_mult

        solver_output = super().solve(dictionary_matrix_scaled, data_work, start_value=start_value)
        solver_output.solution = solver_output.solution / dictionary_scale * data_scale / self.unit_mult
        return solver_output


class L1RegularizedLeastSquaresProblem(LeastSquaresProblem):
    """Least-squares inverse problem with an added L1 regularization term."""

    #: Weight of the L1 regularization term.
    alpha = Range(low=0.0, high=1.0, value=0)

    #: Unique identifier for this problem configuration. (read-only)
    digest = Property(
        depends_on=['solver.digest', 'nonnegative', '_dictionary_scaling', '_data_scaling', 'alpha', 'unit_mult']
    )

    @cached_property
    def _get_digest(self):
        return digest(self)
